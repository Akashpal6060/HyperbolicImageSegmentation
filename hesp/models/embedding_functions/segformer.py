# hesp/models/embedding_functions/segformer.py

from transformers import SegformerForSemanticSegmentation

def segformer_b0(num_labels: int, pretrained_model: str):
    """
    Instantiate SegFormer-B0 for semantic segmentation.
    Args:
      num_labels: number of output channels (tree.M)
      pretrained_model: HF model id or local path
    Returns:
      A SegformerForSemanticSegmentation instance.
    """
    return SegformerForSemanticSegmentation.from_pretrained(
        pretrained_model,
        ignore_mismatched_sizes=True,  # so HF will rebuild the head for num_labels
        num_labels=num_labels,
    )
