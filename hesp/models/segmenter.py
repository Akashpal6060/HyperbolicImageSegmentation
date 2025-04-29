import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

from hesp.hierarchy.tree import Tree
from hesp.config.config import Config
from hesp.embedding_space.embedding_space import EmbeddingSpace
from hesp.util.loss import CCE
from hesp.util.hyperbolic_nn import EPS


class Segmenter(nn.Module):
    """
    PyTorch Segmenter module with SegFormer backbone and hyperbolic embedding head.

    Args:
        tree:        Tree object with class hierarchy.
        config:      Config object.
        train_embedding_space: bool, whether to update embedding-space prototypes.
        prototype_path: path to pretrained prototypes (optional).
    """
    def __init__(
        self,
        tree: Tree,
        config: Config,
        train_embedding_space: bool = True,
        prototype_path: str = ""
    ):
        super().__init__()
        self.tree = tree
        self.config = config
        self.train_embedding_space = train_embedding_space
        self.prototype_path = prototype_path

        # 1) SegFormer backbone
        self.backbone = SegformerForSemanticSegmentation.from_pretrained(
            config.segmenter._PRETRAINED_MODEL,
            ignore_mismatched_sizes=True,
            num_labels=self.tree.K
        )

        # optionally freeze backbone or BN
        if config.segmenter._FREEZE_BACKBONE:
            for param in self.backbone.segformer.encoder.parameters():
                param.requires_grad = False
        if config.segmenter._FREEZE_BN:
            for module in self.backbone.segformer.encoder.modules():
                if isinstance(module, nn.LayerNorm):
                    module.eval()
                    for p in module.parameters():
                        p.requires_grad = False

        # 2) Dimensionality adaptation head (if needed)
        efn_out = config.segmenter._EFN_OUT_DIM
        if efn_out != config.embedding_space._DIM:
            self.down_proj = nn.Conv2d(efn_out, config.embedding_space._DIM, kernel_size=1)
        else:
            self.down_proj = nn.Identity()

        # 3) Hyperbolic embedding-space
        self.embedding_space = EmbeddingSpace(
            tree=self.tree,
            config=self.config,
            train=self.train_embedding_space,
            prototype_path=self.prototype_path
        )


    def forward(self, pixel_values: torch.Tensor) -> dict:
        """
        Args:
            pixel_values: Tensor of shape (B,3,H,W), pre-normalized.
        Returns:
            Dict with 'embeddings', 'predictions', 'probabilities', 'features'.
        """
        # 1) Backbone forward with hidden states enabled
        outputs = self.backbone(pixel_values=pixel_values, output_hidden_states=True)
        logits = outputs.logits  # (B, M, H', W')

        # 2) Safe retrieval of features: last hidden state if present, else logits
        hidden_states = outputs.hidden_states  # tuple of (B, D, H', W')
        features = hidden_states[-1] if hidden_states is not None else logits

        # 3) Adapt to embedding dim
        embeddings = self.down_proj(features)

        # 4) Project to hyperbolic space
        projected = self.embedding_space.project(embeddings)

        # 5) Compute joint & class probabilities
        probs, cprobs = self.embedding_space.run(projected)

        # 6) Decisions (predicted classes)
        preds = self.embedding_space.decide(probs)

        return {
            'embeddings': projected,
            'predictions': preds,
            'probabilities': cprobs,
            'features': pixel_values,
        }



    def compute_loss(self, cprobs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute hierarchical cross-entropy loss (CCE) on valid pixels.

        Args:
            cprobs: Tensor (B,M,H',W') of class probabilities.
            labels: Tensor (B,H_raw,W_raw) of ground-truth indices.
        Returns:
            Scalar loss tensor.
        """
        B, M, Hc, Wc = cprobs.shape
        # resize labels to match cprobs spatial resolution
        labels_resized = F.interpolate(
            labels.unsqueeze(1).float(), size=(Hc, Wc), mode='nearest'
        ).long().squeeze(1)

        # mask out invalid (>M-1)
        valid_mask = labels_resized < M
        valid_cprobs = cprobs.permute(0,2,3,1)[valid_mask]  # (N_valid, M)
        valid_labels = labels_resized[valid_mask]

        # hierarchical cross entropy via embedding_space method
        loss = CCE(valid_cprobs, valid_labels, self.embedding_space.tree)
        # loss = self.embedding_space.cross_entropy(valid_cprobs, valid_labels)
        return loss

    def predict(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Run forward & return upsampled predictions to input size.
        """
        out = self.forward(pixel_values)
        preds = out['predictions']  # (B,Hc,Wc)
        # upsample to original H, W
        B, _, H, W = pixel_values.shape
        preds_up = F.interpolate(
            preds.unsqueeze(1).float(), size=(H, W), mode='nearest'
        ).long().squeeze(1)
        return preds_up
