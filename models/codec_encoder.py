import torch
import torch.nn as nn
from transformers.models.encodec.modeling_encodec import *

class EncodecEncoderQuantizer(EncodecPreTrainedModel):
    def __init__(self, config: EncodecConfig):
        super().__init__(config)
        self.config = config

        self.encoder = EncodecEncoder(config)

        self.quantizer = EncodecResidualVectorQuantizer(config)

        self.bits_per_codebook = int(math.log2(self.config.codebook_size))
        if 2**self.bits_per_codebook != self.config.codebook_size:
            raise ValueError("The codebook_size must be a power of 2.")

        # Initialize weights and apply final processing
        self.post_init()

    def _get_codes_mask(self, codes: torch.Tensor):
        """
        codes: B, 2, N
        """

        firs_quantize = codes[0, :, :] # B, N

        not_62 = firs_quantize != 62

        reversed_not_62 = not_62.flip(dims=[-1])

        cumulative_mask = reversed_not_62.cummax(dim=-1)[0]

        final_mask = cumulative_mask.flip(dims=[-1]).float() # B, N

        # final_mask = final_mask.unsqueeze(-1).expand(-1, -1, self.config.codebook_dim) # B, N, H

        return final_mask

    def forward(
        self,
        input_values: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        bandwidth: Optional[float] = None
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], EncodecOutput]:
        
        if bandwidth is None:
            bandwidth = self.config.target_bandwidths[0]
        if bandwidth not in self.config.target_bandwidths:
            raise ValueError(
                f"This model doesn't support the bandwidth {bandwidth}. "
                f"Select one of {self.config.target_bandwidths}."
            )
        
        if padding_mask is None:
            padding_mask = torch.ones_like(input_values)
        padding_mask = padding_mask.bool()

        length = input_values.shape[-1]
        duration = length / self.config.sampling_rate

        if self.config.chunk_length_s is not None and duration > 1e-5 + self.config.chunk_length_s:
            raise RuntimeError(f"Duration of frame ({duration}) is longer than chunk {self.config.chunk_length_s}")

        scale = None
        if self.config.normalize:
            # if the padding is non zero
            input_values = input_values * padding_mask
            mono = torch.sum(input_values, 1, keepdim=True) / input_values.shape[1]
            scale = mono.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-8
            input_values = input_values / scale

        embeddings = self.encoder(input_values) # B, H, N

        codes = self.quantizer.encode(embeddings, bandwidth) # 2, B, N

        codes_mask = self._get_codes_mask(codes) # B, N, H

        embedded = self.quantizer.decode(codes).transpose(1,2) # B, N, H

        return (embedded, codes_mask)


class CodecEncoder(nn.Module):
    def __init__(
        self,
        codec_pretrained_path: str,
        intermidiate_size: int,
        freeze_codec: bool = True,
        **kwargs
    ):
        super().__init__()
        self.codec_bb = EncodecEncoderQuantizer.from_pretrained(codec_pretrained_path)

        self.hidden_size = self.codec_bb.config.hidden_size
        self.intermidiate_size = intermidiate_size

        self.projector = nn.Linear(self.hidden_size, self.intermidiate_size)

        self.freeze_codec = freeze_codec
        if freeze_codec:
            self.freeze_codec_bb()

    
    def forward(
        self,
        input_values: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        bandwidth: Optional[float] = None,
        **kwargs
    ):
        """
        input_values: B, 1, L
        padding_mask: B, L
        """

        embedded, codes_mask = self.codec_bb(input_values=input_values, padding_mask=padding_mask, bandwidth=bandwidth) # B, N, H

        hidden_states = self.projector(embedded) # B, N, I

        codes_mask = codes_mask.unsqueeze(-1).expand(-1, -1, self.intermidiate_size) # B, N, I

        hidden_states = hidden_states*codes_mask

        pooled_output = hidden_states.sum(dim=1) / codes_mask.sum(dim=1) # B, I

        return pooled_output

    def freeze_codec_bb(self):
        """
        freeze codec backbone
        """
        for param in self.codec_bb.parameters():
            param.requires_grad = False

