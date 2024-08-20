import torch
import torch.nn as nn
from languagecodec_decoder.feature_extractors import EncodecFeatures
import yaml



class CodecEncoder(nn.Module):
    def __init__(
        self,
        codec_pretrained_path: str,
        intermidiate_size: int,
        freeze_codec: bool = False,
        **kwargs
    ):
        self.codec_bb, hidden_size = self.init_codec_bb(codec_pretrained_path)
        self.projector = nn.Linear(hidden_size, intermidiate_size)
        self.freeze_codec = freeze_codec
        if freeze_codec:
            self.freeze_codec_bb()

    
    def forward(
        self,
        audio: torch.Tensor,
        bandwidth_id: torch.Tensor,
        **kwargs
    ):
        quantized, codes, commit_loss = self.codec_bb(audio, bandwidth_id) if self.freeze_codec else self.codec_bb.infer(audio, bandwidth_id)

        quantized = quantized.transpose(-1, -2) # B, N, H

        hidden_states = self.projector(quantized) # B, N, I




    def freeze_codec_bb(self):
        """
        freeze codec backbone
        """
        for param in self.codec_bb.parameters():
            param.requires_grad = False

    def init_codec_bb(pretrained_path: str):
        """
        init codec backbone from pretrained `LanguageCodec`
        """
        with open(pretrained_path + "/config.yaml", "r") as f:
            config = yaml.safe_load(f)

        model = EncodecFeatures(**config['model']['init_args']["feature_extractor"]["init_args"])

        state_dict = torch.load(pretrained_path + "/model.ckpt", map_location="cpu")['state_dict']

        feature_extractor_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('feature_extractor.'):
                feature_extractor_state_dict[k.replace('feature_extractor.', '')] = v

        model.load_state_dict(feature_extractor_state_dict)

        return model, config['model']['init_args']["backbone"]["init_args"]["input_channels"]

