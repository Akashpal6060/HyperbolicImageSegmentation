import torch

from hesp.config.config import Config
from hesp.embedding_space.abstract_embedding_space import AbstractEmbeddingSpace
from hesp.hierarchy.tree import Tree
from hesp.util.layers import euc_mlr  # make sure this is PyTorch‑compatible


class EuclideanEmbeddingSpace(AbstractEmbeddingSpace):
    """
    PyTorch implementation of a flat (Euclidean) embedding space.
    """

    def __init__(
        self,
        tree: Tree,
        config: Config,
        train: bool = True,
        prototype_path: str = ''
    ):
        super().__init__(tree, config, train, prototype_path)
        self.geometry = 'euclidean'
        assert (
            self.geometry == config.embedding_space._GEOMETRY
        ), f"Config geometry '{config.embedding_space._GEOMETRY}' does not match '{self.geometry}'"

        # In Euclidean space, curvature is always zero
        self.curvature = 0.0

    def project(self, embeddings: torch.Tensor, curvature=None) -> torch.Tensor:
        """
        Identity projection: embeddings remain unchanged in Euclidean space.
        """
        return embeddings

    def logits(
        self,
        embeddings: torch.Tensor,
        offsets: torch.Tensor,
        normals: torch.Tensor,
        curvature=None
    ) -> torch.Tensor:
        """
        Compute logits via Euclidean multi-linear regression.
        Delegates to your euc_mlr layer (must be PyTorch).
        """
        # embeddings: (B, D, H, W) or (..., D)
        # offsets:   (M, D), normals: (M, D)
        return euc_mlr(embeddings, P_mlr=offsets, A_mlr=normals)

