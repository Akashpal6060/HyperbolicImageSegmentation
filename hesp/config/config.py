# hesp/config/config.py

import os
import logging
from pprint import pformat
import random

import numpy as np
import torch

from hesp.config.dataset_config import DATASET_CFG_DICT
from hesp.config.embedding_space_config import EmbeddingSpaceConfig
from hesp.config.prototyper_config import VisualizerConfig
from hesp.config.segmenter_config import SegmenterConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Config:
    """
    Central config object for dataset, embedding‐space, segmenter & visualizer.
    Seeds Python, NumPy and PyTorch.
    Builds identifiers & save paths just like the TF version did.
    """

    def __init__(
        self,
        dataset: str,
        mode: str,
        base_save_dir: str = "",
        gpu_idx: int = 0,
        json_selection: str = ""
    ):
        # 1) Mode & base directory
        self._MODE = mode
        self._BASE_DIR = base_save_dir if base_save_dir else os.getcwd()

        # 2) Pick your dataset config
        try:
            # instantiate the class (we trimmed DATASET_CFG_DICT to only 'iddaw')
            self.dataset = DATASET_CFG_DICT[dataset]()
            # override JSON_FILE if user passed one explicitly
            if json_selection and hasattr(self.dataset, "_JSON_FILE"):
                self.dataset._JSON_FILE = os.path.join(
                    self.dataset._DATASET_DIR, json_selection
                )
        except KeyError:
            valid = ", ".join(DATASET_CFG_DICT.keys())
            logger.error(f"Dataset '{dataset}' not supported. Valid options: {valid}")
            raise NotImplementedError

        # 3) GPU index (if you’ll use it later for CUDA device)
        self._GPU_IDX = gpu_idx

        # 4) Seed everything
        seed = 1
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # 5) Sub‐configs
        self.visualizer      = VisualizerConfig()
        self.segmenter       = SegmenterConfig()
        self.embedding_space = EmbeddingSpaceConfig()


    @property
    def identifier(self) -> str:
        """
        e.g. 'hierarchical_iddaw_d256_hyperbolic_c0.1'
        """
        parts = []
        if getattr(self.embedding_space, "_HIERARCHICAL", False):
            parts.append("hierarchical")

        geom = self.embedding_space._GEOMETRY.lower()
        parts.append(f"{self.dataset._NAME}_d{self.embedding_space._DIM}_{geom}")

        if geom == "hyperbolic":
            parts[-1] += f"_c{self.embedding_space._INIT_CURVATURE}"

        return "_".join(parts)


    @property
    def segment_identifier(self) -> str:
        """
        Builds on top of `identifier`, adding:
         - os<stride>
         - backbone name
         - bs<batch_size>
         - lr<learning_rate>
         - fbn<freeze_bn>
         - fbb<freeze_backbone>
         - optional segmenter_ident
         - '_nomatch' if embedding dim ≠ EFN_OUT_DIM
        """
        ident = self.identifier

        if getattr(self.segmenter, "_ZERO_LABEL", False):
            ident += "_ZL"

        ident += f"_os{self.segmenter._OUTPUT_STRIDE}"
        ident += f"_{self.segmenter._BACKBONE}"
        ident += f"_bs{self.segmenter._BATCH_SIZE}"
        ident += f"_lr{self.segmenter._INITIAL_LEARNING_RATE}"
        ident += f"_fbn{self.segmenter._FREEZE_BN}"
        ident += f"_fbb{self.segmenter._FREEZE_BACKBONE}"

        seg_id = getattr(self.segmenter, "_SEGMENTER_IDENT", "")
        if seg_id:
            ident += f"_{seg_id}"

        if (
            getattr(self.embedding_space, "_DIM", None)
            != getattr(self.segmenter, "_EFN_OUT_DIM", None)
        ):
            ident += "_nomatch"

        return ident


    @property
    def segmenter_save_dir(self) -> str:
        """
        Mirrors the TF:

          home/poincare-hesp/save/<segmenter_dir>/<segment_identifier>/
        """
        save_dir = os.path.join(
            self._BASE_DIR,
            "poincare-hesp",
            "save",
            self.segmenter._SEGMENTER_DIR,
            self.segment_identifier,
        )
        os.makedirs(save_dir, exist_ok=True)
        return save_dir


    def pretty_print(self):
        """
        Logs a nice summary of the configs and save‐dirs.
        """
        cfg = {
            "dataset":         {k: v for k, v in vars(self.dataset).items() if not k.startswith("__")},
            "embedding_space": vars(self.embedding_space),
            "save_dirs":       {},
        }

        if self._MODE == "segmenter":
            cfg["segmenter"]            = vars(self.segmenter)
            cfg["save_dirs"]["segmenter"] = self.segmenter_save_dir

        logger.info("Configuration:\n%s", pformat(cfg))
