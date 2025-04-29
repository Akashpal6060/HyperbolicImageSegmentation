# hesp/models/model_factory.py

import json
import logging
from pprint import pformat

import torch

from hesp.hierarchy.tree import Tree
from hesp.models.segmenter import Segmenter
from hesp.config.config import Config

logger = logging.getLogger(__name__)


class ModelFactory:
    @staticmethod
    def create(model_name: str, cfg: Config) -> torch.nn.Module:
        """
        Initializes and returns a model based on `model_name` and `cfg`.
        """
        # 1) Pretty‐print configuration
        logger.info("CONFIGURATION")
        cfg.pretty_print()

        # 2) Load or skip the hierarchy JSON
        hierarchy_json = {}
        if getattr(cfg.embedding_space, "_HIERARCHICAL", False):
            with open(cfg.dataset._JSON_FILE, "r") as f:
                hierarchy_json = json.load(f)

        # 3) Build the class‐tree
        tree = Tree(cfg.dataset._I2C, hierarchy_json)
        logger.info("Hierarchy structure:\n%s", pformat(tree.json))

        # 4) Choose & instantiate the model
        if model_name == "segmenter":
            model = Segmenter(
                tree=tree,
                config=cfg,
                train_embedding_space=True,
                prototype_path=""
            )
        else:
            raise ValueError(f"Unsupported model: {model_name!r}")

        # 5) Move to device
        device = (
            torch.device(f"cuda:{cfg._GPU_IDX}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        model.to(device)
        logger.info(f"Moved model to device: {device}")

        return model
