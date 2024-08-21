from transformers.models.wav2vec2.modeling_wav2vec2 import *


class SemanticEncoder(nn.Module):
    def __init__(
        self,
        semantic_pretrained_path: str,
        intermidiate_size: int,
        freeze_semantic: bool = False,
        **kwargs
    ):
        super().__init__()
        self.semantic_bb = Wav2Vec2Model.from_pretrained(semantic_pretrained_path)
        config = self.semantic_bb.config
        self.projector = nn.Linear(config.hidden_size, intermidiate_size)
        if freeze_semantic:
            self.freeze_semantic_bb()

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        **kwargs
    ):

        outputs = self.semantic_bb(
            input_features,
            attention_mask=attention_mask,
            **kwargs
        )

        hidden_states = outputs[0] # B, N, H

        hidden_states = self.projector(hidden_states) # B, N, I

        pooled_output = hidden_states.mean(dim=1)

        # if attention_mask is None:
        #     pooled_output = hidden_states.mean(dim=1) # B, I
        # else:
        #     padding_mask = self.semantic_bb._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)
        #     hidden_states[~padding_mask] = 0.0
        #     pooled_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1) # B, I

        return pooled_output
        
    def freeze_semantic_bb(self):
        """
        freeze semantic backbone
        """
        for param in self.semantic_bb.parameters():
            param.requires_grad = False

