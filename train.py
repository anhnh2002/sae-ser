import torch
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
import numpy as np
from models.speech_emotion_recognizer import SER
from utils.configs import SERConfig
from utils.custom_dataset import CustomDataset, DataCollatorForSER

from transformers.models.seamless_m4t.feature_extraction_seamless_m4t import SeamlessM4TFeatureExtractor
from transformers.models.encodec.feature_extraction_encodec import EncodecFeatureExtractor

import os
from tqdm import tqdm
import json
import wandb
import pandas as pd
from sklearn.model_selection import train_test_split

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
    train_anot = pd.read_csv(f"{data_root}/anots/{fold}.train.csv")
    X = train_anot.drop('emotion', axis=1)  # Lấy tất cả các cột ngoại trừ cột mục tiêu
    y = train_anot['emotion']  
    X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

    train_anot = pd.concat([X_train, y_train], axis=1)
    eval_anot = pd.concat([X_eval, y_eval], axis=1)

    test_anot = pd.read_csv(f"{data_root}/anots/{fold}.test.csv")

    train_dataset = CustomDataset(train_anot, f"{data_root}/wavs/")
    eval_dataset = CustomDataset(eval_anot, f"{data_root}/wavs/")
    test_dataset = CustomDataset(test_anot, f"{data_root}/wavs/")
    print(len(train_dataset), len(eval_dataset), len(test_dataset))


    freeze_codec = False
    freeze_semantic = False

    # Initialize the model
    config = SERConfig(
        num_labels=4,
        codec_pretrained_path = "facebook/encodec_24khz",
        freeze_codec = freeze_codec,
        semantic_pretrained_path = "facebook/w2v-bert-2.0",
        freeze_semantic = freeze_semantic,
        merge_strategy="concat",
        intermidiate_size=512,
        dropout=0.1,
    )

    model = SER(config).to("cuda")

    out_dir = f"{output_root}/{fold}-freeze_codec-{freeze_codec}-freeze_semantic-{freeze_semantic}"
    os.makedirs(out_dir, exist_ok=True)

    #args
    lr = 5e-4
    epochs = 5

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=out_dir,
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,
        per_device_eval_batch_size=32,
        warmup_steps=100,
        weight_decay=0.001,
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to=None
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
    # wandb.login(key='e0e0a2547f255a36f551d7b6a166b84e5139d276')
    # wandb.init(
    #     project="Speech Emotion Recognition",
    #     name="wav2vec2-bert_codec",
    #     config={
    #             "learning_rate": lr,
    #             "epochs": epochs,
    #             "dataset": "IEMOCAP"
    #         }
    # )

    # Train the model
    trainer.train()

    # Save the best model
    best_model_path = f"{training_args.output_dir}/best_model"
    trainer.save_model(best_model_path)

    torch.cuda.empty_cache()

    # Test
    res = trainer.predict(test_dataset=test_dataset)._asdict()
    res["predictions"] = res["predictions"].tolist()
    res["label_ids"] = res["label_ids"].tolist()
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