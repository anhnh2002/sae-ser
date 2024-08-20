import torch
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
import numpy as np
from models.speech_emotion_recognizer import SER
from utils.configs import SERConfig
from utils.custom_dataset import CustomDataset, DataCollatorForSER
from torch.utils.data import DataLoader
from transformers.models.seamless_m4t.feature_extraction_seamless_m4t import SeamlessM4TFeatureExtractor
from transformers.models.encodec.feature_extraction_encodec import EncodecFeatureExtractor

import os
from tqdm import tqdm
import json
import wandb


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

def train(
    fold: str,
    data_root: str = "datasets/iemocap",
    output_root: str = "checkpoints/ser"
):
    
    os.makedirs(output_root, exist_ok=True)
    
    # Init dataset
    train_dataset = CustomDataset(f"{data_root}/anots/{fold}.train.csv")
    test_dataset = CustomDataset(f"{data_root}/anots/{fold}.test.csv", shuffle=False)
    split_ = int(0.9*len(train_dataset))
    eval_dataset = train_dataset[split_:]
    train_dataset = train_dataset[:split_]

    # Initialize the model
    config = SERConfig(
        num_labels=4,
        codec_pretrained_path = "facebook/encodec_24khz",
        freeze_codec = True,
        semantic_pretrained_path = "facebook/w2v-bert-2.0",
        freeze_semantic = False,
        merge_strategy="concat",
        intermidiate_size=512,
        dropout=0.1,
    )

    model = SER(config)

    os.makedirs(f"{output_root}/{fold}", exist_ok=True)

    #args
    lr = 2e-5
    epochs = 5

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=f"{output_root}/{fold}",
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to="wandb"
    )

    codec_processor = EncodecFeatureExtractor.from_pretrained(config.codec_pretrained_path)
    semantic_processor = SeamlessM4TFeatureExtractor.from_pretrained(config.semantic_pretrained_path)

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorForSER(codec_processor=codec_processor, semantic_processor=semantic_processor),  # Implement this data collator
    )

    # wandb init
    wandb.login(key='e0e0a2547f255a36f551d7b6a166b84e5139d276')
    wandb.init(
        project="Speech Emotion Recognition",
        name="wav2vec2-bert_codec",
        config={
                "learning_rate": lr,
                "epochs": epochs,
                "dataset": "IEMOCAP"
            }
    )

    # Train the model
    trainer.train()

    # Save the best model
    best_model_path = f"{training_args.output_dir}/best_model"
    trainer.save_model(best_model_path)

    # Test
    res = trainer.predict(test_dataset=test_dataset)._asdict()
    os.makedirs(f"results/{fold}", exist_ok=True)
    with open(f"results/{fold}/result.json", 'w') as json_file:
        json.dump(res, json_file, indent=4)

    print("\n#######################################################")
    print(f"Test result on {fold}: {res['metrics']}")
    print("#######################################################\n")



if __name__ == "__main__":
    train(
        fold="iemocap_01F",
        data_root="datasets/iemocap",
        output_root="checkpoints/ser"
    )