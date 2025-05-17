import torch
import argparse
import os
import json
import logging 
import pandas as pd
import torch
import math
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from tqdm import tqdm
from dotenv import load_dotenv, dotenv_values 
# loading variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

POLITIFACT_LABELS = {
    'true': 0,
    'half-true': 1,
    'false': 2,
}

class ClaimDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.claims = [claim for claim in dataframe['claim'].iloc()]
        self.evidences = ['.'.join(evidences) for evidences in dataframe['fact_checking_evidence'].iloc()]
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

def evaluate_model(args):
    SAVE_PATH = args.save_path
    MODEL_WEIGHTS_PATH = args.model_weights_path
    EVAL_DATA_PATH = args.eval_data_path
    NUM_LABELS = args.num_labels
    EVAL_BATCH_SIZE = args.eval_batch_size
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(SAVE_PATH):
        logging.info("Creating Save Path")
        os.makedirs(SAVE_PATH)
        
    # Save current args to output
    logging.info("Saving CLI stated config")
    with open(f"{SAVE_PATH}/argsparse_config.json", 'w') as file:
        json.dump(vars(args), file)
        file.close()
        
    # Load Model Weights
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_WEIGHTS_PATH, num_labels=NUM_LABELS).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_WEIGHTS_PATH)
    
    # Load Dataset
    eval_data = pd.read_json(EVAL_DATA_PATH)
    eval_dataset = ClaimDataset(eval_data, tokenizer)
    eval_dataloader = DataLoader(eval_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=True)
    
    model.eval()
    evals = {
        'eval_loss': [],
        'eval_preds': [],
        'eval_true': []
    }
    with torch.no_grad():
        total_eval_loss = []
        total_eval_preds = []
        total_eval_true = []
        eval_progress_bar = tqdm(eval_dataloader)
        
        for batch in eval_progress_bar:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_masks = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_masks, labels=labels)
            loss = outputs.loss
            total_eval_loss.append(loss.item())
            predictions = torch.argmax(outputs.logits, dim=1)
            preds_data = [str(x) for x in predictions.cpu().tolist()]
            total_eval_preds.append(','.join(preds_data))
            true_data = [str(x) for x in labels.cpu().tolist()]
            total_eval_true.append(','.join(true_data))
                
            eval_accuracy = (predictions == labels).sum().item() / labels.size(0)
            eval_progress_bar.set_postfix(loss=outputs.loss.item(), accuracy=eval_accuracy)
        
        evals['eval_loss'].append(total_eval_loss)
        evals['eval_preds'].append(total_eval_preds)
        evals['eval_true'].append(total_eval_true)
        
    dataframe = pd.DataFrame(evals)
    dataframe.to_json(os.path.join(SAVE_PATH, 'eval.json'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine Tuning Models")
    # Data Arguments
    parser.add_argument("--save_path", type=str, default=os.path.join('script_outputs', 'finetuned_encoder_models_results', 'bert-base-uncased', 'round_1'))
    parser.add_argument("--model_weights_path", type=str, default=os.path.join('script_outputs', 'finetuned_encoder_models', 'bert-base-uncased'))
    parser.add_argument("--eval_data_path", type=str, default=os.path.join('script_outputs', 'role_playing_misinformation_generated_datasets', '2025-03-21-00-48-13', 'round_1.json'))
    parser.add_argument("--num_labels", type=int, default=3)
    
    # Eval Arguments
    parser.add_argument("--eval_batch_size", type=int, default=8)
    
    args = parser.parse_args()
    evaluate_model(args)