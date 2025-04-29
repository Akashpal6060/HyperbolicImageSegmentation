import os
import numpy as np
import torch
import torch.nn as nn

from hesp.config.config import Config
from hesp.embedding_space.abstract_embedding_space import AbstractEmbeddingSpace
from hesp.hierarchy.tree import Tree
from hesp.util.hyperbolic_nn import exp_map_zero  # PyTorch implementation
from hesp.util.layers import hyp_mlr  # PyTorch implementation


class HyperbolicEmbeddingSpace(AbstractEmbeddingSpace):
    """
    PyTorch implementation of a hyperbolic embedding space.

    Uses Poincaré exponential map at the origin and a hyperbolic multi-linear regression head.
    """
    def __init__(
        self,
        tree: Tree,
        config: Config,
        train: bool = True,
        prototype_path: str = ''
    ):
        super().__init__(tree, config, train, prototype_path)
        self.geometry = 'hyperbolic'
        assert (
            self.geometry == config.embedding_space._GEOMETRY
        ), f"Config geometry '{config.embedding_space._GEOMETRY}' does not match '{self.geometry}'"

        # Initialize curvature parameter
        if not train and prototype_path:
            c_npy = np.load(os.path.join(prototype_path, 'c.npy'))
            init_c = float(c_npy)
        else:
            init_c = config.embedding_space._INIT_CURVATURE

        self.curvature = nn.Parameter(
            torch.tensor(init_c, dtype=torch.float32), requires_grad=train
        )

    def project(
        self,
        embeddings: torch.Tensor,
        curvature: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Maps Euclidean embeddings onto the Poincaré ball via exponential map at origin.

        embeddings: (B, D, H, W) or (..., D)
        curvature: scalar or tensor representing c
        """
        c = curvature if curvature is not None else self.curvature
        return exp_map_zero(embeddings, c=c)

    def logits(
        self,
        embeddings: torch.Tensor,
        offsets: torch.Tensor,
        normals: torch.Tensor,
        curvature: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute hyperbolic logits via hyperbolic multi-linear regression.

        Returns a tensor of shape (..., M).
        """
        c = curvature if curvature is not None else self.curvature
        return hyp_mlr(
            embeddings,
            c=c,
            P_mlr=offsets,
            A_mlr=normals
        )
