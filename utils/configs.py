from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers.models.encodec.configuration_encodec import EncodecConfig
from transformers.models.wav2vec2_bert.configuration_wav2vec2_bert import Wav2Vec2BertConfig


logger = logging.get_logger(__name__)


class SERConfig(PretrainedConfig):
    model_type = "heterogeneous"

    def __init__(
        self,
        num_labels: int,
        codec_pretrained_path: str = "facebook/encodec_24khz",
        freeze_codec: bool = True,
        semantic_pretrained_path: str = "facebook/w2v-bert-2.0",
        freeze_semantic: bool = False,
        intermidiate_size: int = 512,
        merge_strategy: str = "concat",
        dropout: float = 0.2,
        **kwargs,
    ):
        super(Wav2Vec2BertConfig, self).__init__(num_labels=num_labels, **kwargs)
        self.codec_pretrained_path = codec_pretrained_path
        self.freeze_codec = freeze_codec
        self.semantic_pretrained_path = semantic_pretrained_path
        self.freeze_semantic = freeze_semantic
        self.intermidiate_size = intermidiate_size
        self.merge_strategy = merge_strategy
        self.dropout = dropout
        self.codec_config = EncodecConfig.from_pretrained(codec_pretrained_path)
        self.semantic_config = Wav2Vec2BertConfig.from_pretrained(semantic_pretrained_path)
    
    