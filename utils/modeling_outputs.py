from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from transformers.utils import ModelOutput

@dataclass
class SEROutput(ModelOutput):
    """
    Base class for outputs of sentence classification models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None