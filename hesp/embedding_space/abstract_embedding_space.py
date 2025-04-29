import logging
import os
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
from hesp.config.config import Config
from hesp.hierarchy.tree import Tree

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AbstractEmbeddingSpace(nn.Module, ABC):
    """
    Base class for embedding spaces in PyTorch.
    Supports hierarchical softmax when the tree is hierarchical.
    """
    def __init__(
        self,
        tree: Tree,
        config: Config,
        train: bool = True,
        prototype_path: str = ''
    ):
        super().__init__()
        self.tree = tree
        self.config = config
        self.dim = config.embedding_space._DIM
        self.hierarchical = config.embedding_space._HIERARCHICAL
        self.curvature = config.embedding_space._INIT_CURVATURE

        # Determine number of classes M
        self.M = self.tree.M if self.hierarchical else self.tree.K
        # Initialize or load prototypes
        if not train and prototype_path:
            normals_npy = np.load(os.path.join(prototype_path, 'normals.npy'))
            offsets_npy = np.load(os.path.join(prototype_path, 'offsets.npy'))
            normals = torch.tensor(normals_npy, dtype=torch.float32)
            offsets = torch.tensor(offsets_npy, dtype=torch.float32)
        else:
            normals = torch.randn(self.M, self.dim) * 0.05
            offsets = torch.zeros(self.M, self.dim)

        # Register parameters
        self.normals = nn.Parameter(normals, requires_grad=train)
        self.offsets = nn.Parameter(offsets, requires_grad=train)

        # Build hierarchy buffers of correct size
        full_hmat = torch.tensor(self.tree.hmat, dtype=torch.float32)
        full_sib = torch.tensor(self.tree.sibmat, dtype=torch.float32)
        if self.hierarchical:
            hmat = full_hmat
            sibmat = full_sib
        else:
            # flat: only leaves
            hmat = full_hmat[:self.M, :self.M]
            sibmat = full_sib[:self.M, :self.M]

        # Register hierarchy matrices as buffers
        self.register_buffer('hmat', hmat)
        self.register_buffer('sibmat', sibmat)

    def softmax(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Hierarchical (or flat) softmax: cond_probs of shape [B, M, H, W]
        """
        B, M_out, H, W = logits.shape
        assert M_out == self.M, f"Expected logits with {self.M} channels, got {M_out}"

        # Stability fix: subtract max per spatial location
        max_per_loc = logits.amax(dim=1, keepdim=True)         # [B, 1, H, W]
        shifted = logits - max_per_loc                         # [B, M, H, W]
        exp_logits = shifted.exp()                             # [B, M, H, W]

        # Flatten spatial dimensions for batched matrix multiplication
        flat = exp_logits.permute(0, 2, 3, 1).reshape(-1, self.M)  # [N, M]

        # Z_flat: Normalizer using sibling matrix
        Z_flat = flat @ self.sibmat                              # [N, M]
        
        # Conditional probabilities normalized over siblings
        cond_flat = flat / Z_flat.clamp(min=1e-15)               # [N, M]

        # Reshape back to original spatial structure
        cond_probs = cond_flat.view(B, H, W, self.M).permute(0, 3, 1, 2)  # [B, M, H, W]

        return cond_probs

    def decide(self, probs: torch.Tensor, unseen: list = []) -> torch.Tensor:
        """
        Decide leaf class from joint probabilities. Returns shape [B, H, W].
        """
        K = self.tree.K
        probs_leaf = probs[:, :K, :, :]                               # [B,K,H,W]

        B, _, H, W = probs_leaf.shape
        flat = probs_leaf.permute(0, 2, 3, 1).reshape(-1, K)         # [N, K]

        if unseen:
            device = flat.device
            unseen_idx = torch.tensor(unseen, dtype=torch.long, device=device)
            gathered = flat[:, unseen_idx]                            # [N, len(unseen)]
            choice = gathered.argmax(dim=1)                            # [N]
            pred_flat = unseen_idx[choice]
        else:
            pred_flat = flat.argmax(dim=1)                            # [N]

        preds = pred_flat.reshape(B, H, W)
        return preds

    def run(
        self,
        embeddings: torch.Tensor,
        offsets: torch.Tensor = None,
        normals: torch.Tensor = None,
        curvature: float = None
    ) -> tuple:
        """
        Computes joint and conditional probabilities from embeddings.
        Returns: (joints [B,M,H,W], cond_probs [B,M,H,W])
        """
        offsets = offsets if offsets is not None else self.offsets
        normals = normals if normals is not None else self.normals
        curvature = curvature if curvature is not None else self.curvature

        logits = self.logits(embeddings, offsets, normals, curvature)  # [B,M,H,W]
        cond_probs = self.softmax(logits)
        joints = self.get_joints(cond_probs)
        return joints, cond_probs

    def get_joints(self, cond_probs: torch.Tensor) -> torch.Tensor:
        """
        Calculates joint probabilities: exp(log_probs @ hmat).
        Returns [B,M,H,W].
        """
        B, M_out, H, W = cond_probs.shape
        assert M_out == self.M

        log_flat = torch.log(cond_probs.clamp(min=1e-4)).permute(0, 2, 3, 1).reshape(-1, self.M)  # [N,M]
        log_sum_flat = log_flat.matmul(self.hmat)                                                    # [N,M]

        joints = log_sum_flat.exp().reshape(B, H, W, self.M).permute(0, 3, 1, 2)                    # [B,M,H,W]
        return joints

    @abstractmethod
    def logits(
        self,
        embeddings: torch.Tensor,
        offsets: torch.Tensor,
        normals: torch.Tensor,
        curvature: float
    ) -> torch.Tensor:
        """
        Given embeddings, compute logits for each class.
        Must return shape [B, M, H, W].
        """
        pass