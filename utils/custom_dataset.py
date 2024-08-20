from torch.utils.data import Dataset
from transformers import AutoFeatureExtractor
from transformers.models.encodec.feature_extraction_encodec import EncodecFeatureExtractor

class CustomDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        codec_pretrained_path: str = "facebook/encodec_24khz",
        semantic_pretrained_path: str = "facebook/w2v-bert-2.0"
    ):
        super().__init__()
        self.codec_processor = EncodecFeatureExtractor.from_pretrained(codec_pretrained_path)
        self.semantic_processor = AutoFeatureExtractor.from_pretrained(semantic_pretrained_path)
        self._init_data(data_dir)

    def _init_data(self, data_dir):
        pass

    def __len__(self):
        return len(self.anot)

    def __getitem__(self, idx):
        sample = self.anot.iloc[idx]
        fn = sample.filename

        input_ids = self.tokenizer(str(sample.label), return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_seq_len)

        img_path = "augmented_images/" + fn
        if fn[:4] == "wild":
            img_path = "WildLine/" + fn
        if fn[:5] == "digit":
            img_path = "digits/" + fn
        if fn[:6] == "single":
            img_path = "single_digit/" + fn

        # return {
        #     "pixel_values":    pixel_values[0],
        #     "input_ids":   input_ids["input_ids"][0],
        #     "att_mask":     input_ids["attention_mask"][0]
        # }