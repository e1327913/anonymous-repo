import argparse
import os
import json
import logging
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# importing necessary functions from dotenv library
from dotenv import load_dotenv, dotenv_values 
# loading variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

POLITIFACT_LABELS = {
    'true': 0,
    'half-true': 1,
    'false': 2
}

class ClaimDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.claims = dataframe['claim'].iloc()
        self.evidences = ['.'.join(evidences) for evidences in dataframe['fact_checking_evidences'].iloc()]
        self.labels = [POLITIFACT_LABELS[label] for label in dataframe['label'].iloc()]
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        claim = self.claims[index]
        evidences = self.evidences[index]
        label = self.labels[index]
        
        inputs = self.tokenizer(claim, evidences, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
        
        inputs["labels"] = torch.tensor(label, dtype=torch.long)
        
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": inputs["labels"]
        }

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="macro")
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# Function to Load Dataset from JSON
def load_dataset_from_json(path):
    df = pd.read_json(path)
    df['label'] = df['label'].map(POLITIFACT_LABELS)
    df['fact_checking_evidences'] = df['fact_checking_evidences'].apply(lambda x: " ".join(x))  # Convert evidence list to string
    df = df.get(['claim', 'fact_checking_evidences', 'label'])
    return Dataset.from_pandas(df)

# Tokenization Function
def tokenize_function(examples, tokenizer):
    return tokenizer(examples["claim"], examples["fact_checking_evidences"], truncation=True, padding="max_length", max_length=512)


def finetune_model(args):
    SAVE_PATH = args.save_path
    TRAIN_DATA_PATH = args.training_data_path
    DEV_DATA_PATH = args.dev_data_path
    NUM_LABELS = args.num_labels
    MODEL_NAME = args.model_name
    NUM_EPOCHS = args.num_epochs
    LEARNING_RATE = args.learning_rate
    TRAIN_BATCH_SIZE = args.train_batch_size
    DEV_BATCH_SIZE = args.dev_batch_size
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(SAVE_PATH):
        logging.info("Creating Save Path")
        os.makedirs(SAVE_PATH)
    # Save current args to output
    logging.info("Saving CLI stated config")
    with open(f"{SAVE_PATH}/argsparse_config.json", 'w') as file:
        json.dump(vars(args), file)
        file.close()

    # Load Model & Tokenizer
    logging.info("Prepare model and tokenizer")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Load Dataset
    train_dataset = load_dataset_from_json(TRAIN_DATA_PATH)
    dev_dataset = load_dataset_from_json(DEV_DATA_PATH)

    train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    dev_dataset = dev_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    dev_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    
    # Declare Training Arguments
    training_args = TrainingArguments(
        output_dir=SAVE_PATH,
        evaluation_strategy="epoch",
        save_strategy="epoch",  # Save model at each epoch
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=DEV_BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.1,  # Enables L2 Regularization
        logging_dir=f"{SAVE_PATH}/logs",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",  # Save best model based on accuracy
        logging_steps=1000,
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    logging.info("Starting Training...")
    trainer.train()

    # Save Final Model
    trainer.save_model(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    logging.info(f"Model saved at: {SAVE_PATH}")

    # Evaluate the Model
    logging.info("Evaluating Model...")
    eval_results = trainer.evaluate()
    logging.info(f"Evaluation Results: {eval_results}")

    # Save Evaluation Metrics
    with open(f"{SAVE_PATH}/eval_results.json", 'w') as f:
        json.dump(eval_results, f)
    
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Fine Tuning Models")
    # Data Arguments
    parser.add_argument("--save_path", type=str, default=os.path.join('script_outputs', 'finetuned_encoder_models', 'bert-large-uncased'))
    parser.add_argument("--training_data_path", type=str, default=os.path.join('script_outputs', 'train_dev_test_split', 'train.json'))
    parser.add_argument("--dev_data_path", type=str, default=os.path.join('script_outputs', 'train_dev_test_split', 'dev.json'))
    parser.add_argument("--num_labels", type=int, default=3)
    
    # Training Arguments
    parser.add_argument("--model_name", type=str, default='bert-large-uncased')
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--dev_batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=5)
    
    args = parser.parse_args()
    finetune_model(args)