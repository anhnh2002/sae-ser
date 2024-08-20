import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel
from codec_encoder import CodecEncoder
from semantic_encoder import SemanticEncoder
from typing import Optional
import torch.nn.functional as F
import json
from utils.configs import SERConfig
from utils.modeling_outputs import SEROutput


class SER(PreTrainedModel):
    config_class = SERConfig

    def __init__(
        self,
        config: SERConfig
    ):
        super(SER, self).__init__(config)

        self.config = config

        self.semantic_encoder = SemanticEncoder(**self.config)

        self.codec_encoder = CodecEncoder(**self.config)

        if self.config.merge_strategy == "concat":
            self.projector = nn.Linear(self.config.intermidiate_size*2, self.config.intermidiate_size*2)
        elif self.config.merge_strategy == "sum":
            self.projector = nn.Linear(self.config.intermidiate_size, self.config.intermidiate_size*2)
        else:
            raise(Exception(f"{self.config.merge_strategy} not be supported. Supported merge strategies: [`sum`, `concat`]"))
        
        self.dropout = nn.Dropout(self.config.dropout)

        self.classifier = nn.Linear(self.config.intermidiate_size*2, self.config.num_labels)

    @staticmethod
    def load_from_checkpoint(self, output_dir: str) -> None:
        with open(output_dir + '/config.json', 'r') as json_file:
            self.config = json.load(json_file)

        model = SER(**self.config)

        model.load_state_dict(torch.load(output_dir + "/model.pth"))

        return model

    def save_checkpoint(self, output_dir: str) -> None:
        torch.save(self.state_dict(), output_dir + "/model.pth")
        with open(output_dir + '/config.json', 'w') as json_file:
            json.dump(self.config, json_file, indent=4)
    
    def forward(
        self,
        input_values: torch.Tensor,
        input_features: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        semantic_reps = self.semantic_encoder(input_features, attention_mask) # B, I
        codec_reps = self.codec_encoder(input_values, padding_mask) # B, I

        if self.config.merge_strategy == "concat":
            merged_reps = torch.cat([semantic_reps, codec_reps], dim=-1) # B, 2I
        else:
            merged_reps = semantic_reps + codec_reps # B, I

        projected = self.dropout(F.relu(self.projector(merged_reps))) # B, 2I

        logits = self.classifier(projected)

        loss = None
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        return SEROutput(logits=logits, loss=loss)