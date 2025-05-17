import os
import argparse
import logging
import torch
import json
import transformers
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import login
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from utils.pytorch_dataset_classes.misinformation_dataset import MisinformationDataset
from utils.huggingface.functions import step, format_chunks

# Load variables from .env file
load_dotenv()
# Prepare Logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def reasoning_prompt(claim, evidence):
    prompt = f"""You are a fact-checking assistant. Your task is to analyze the claim and compare it with the provided evidence.

Claim:  
"{claim}"

Evidence:  
"{evidence}"

Instructions:
1. Carefully analyze whether the claim is fully, partially, or not supported by the evidence.
2. Identify specific factual elements in the claim that are supported or contradicted.
3. Note any missing context, exaggerations, or misleading aspects of the claim.
4. Do not make assumptions beyond what the evidence explicitly states.

"""

    return prompt

def labelling_prompt():
    prompt = f"""Based on your factual analysis, assign the appropriate label.

Label Definitions:
- True: A statement is fully accurate.
- Half-True: A statement that conveys only part of the truth, especially one used deliberately in order to mislead someone.
- False: A statement is inaccurate or contradicted by evidence.

Instructions:
- Avoid assumptions beyond what the analysis states.
- Ensure consistency between the label and reasoning.
- Provide your confidence score for all of the labels.
"""

    return prompt

def average_prompt():
    prompt = f"""Select the label based on the highest confidence score and provide an explanation on your factual analysis.

Output Format (JSON)
```json
{{
  "Label": "<True / Half-True / False>",
  "Explanation": "<Short justification referencing your factual analysis>"
}}
"""

    return prompt

def collate_fn(batch):
    return batch  # Simply return the list of dictionaries

def execute_prompts(histories, batch_size, model, tokenizer, device, max_generated_tokens, temperature):
    outputs = []
    # Tokenize Prompts
    prompts = tokenizer.apply_chat_template(conversation=histories, tokenize=False, add_generation_prompt=True)
     # Load into dataset
    dataset = MisinformationDataset(prompts)
    dataloader = DataLoader(dataset, batch_size, shuffle=False, drop_last=False)
    model.eval()
    with torch.no_grad(): 
        for batch_histories in tqdm(dataloader):
            batch_histories = tokenizer(batch_histories, padding=True, return_tensors="pt").to(device)
            hf_outputs = step(batch_histories, model=model, tokenizer=tokenizer, max_generated_tokens=max_generated_tokens, temperature=temperature)
            outputs += hf_outputs
        for idx, output in enumerate(outputs):
            histories[idx] = format_chunks(output)
    
    return histories

def run_labelling(dataframe, batch_size, model, tokenizer, device, max_generated_tokens, temperature, save_path):
    # Prepare Base Prompts
    histories = []
    reason_path = os.path.join(save_path, f"reason.json")
    if not os.path.exists(reason_path):
        for row in tqdm(dataframe.iloc()):
            claim = row['generated_claims']['Claim'].strip()
            evidence = ' '.join(row['fact_checking_evidences'])
            prompt = reasoning_prompt(claim, evidence)
            histories.append([{'role': 'user', 'content': prompt}])

        histories = execute_prompts(histories, batch_size, model, tokenizer, device, max_generated_tokens, temperature)
        with open(reason_path, 'w') as f:
            json.dump(histories, f)

    labelling_path = os.path.join(save_path, f'labelling.json')
    if not os.path.exists(labelling_path):
        with open(reason_path, 'r') as f:
            histories = json.load(f)

        for idx, row in tqdm(enumerate(dataframe.iloc())):
            prompt = labelling_prompt()
            histories[idx].append({'role': 'user', 'content': prompt})

        histories = execute_prompts(histories, batch_size, model, tokenizer, device, max_generated_tokens, temperature)
        with open(labelling_path, 'w') as f:
            json.dump(histories, f)

    average_path = os.path.join(save_path, f'average.json')
    if not os.path.exists(average_path):
        with open(labelling_path, 'r') as f:
            histories = json.load(f)

        for idx, row in tqdm(enumerate(dataframe.iloc())):
            prompt = average_prompt()
            histories[idx].append({'role': 'user', 'content': prompt})

        histories = execute_prompts(histories, batch_size, model, tokenizer, device, max_generated_tokens, temperature)
        with open(average_path, 'w') as f:
            json.dump(histories, f)

    if os.path.exists(average_path):
        with open(average_path, 'r') as f:
            histories = json.load(f)

    # Return all generated claims
    dataframe['label_histories'] = histories

    return dataframe

def perform_role_playing_labelling(args):
    # Argparse
    DEVICE = args.device
    HF_MODEL = args.hf_model
    SAVE_PATH = args.save_path
    DATASET_PATH = args.dataset_path
    BATCH_SIZE = args.batch_size
    LLM_MAX_TOKENS = args.max_tokens
    LLM_TEMPERATURE = args.temperature

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    # Save current args to output
    with open(f"{SAVE_PATH}/argsparse_config.json", 'w') as file:
        json.dump(vars(args), file)
        file.close()

    # Login to HF 
    login(os.getenv("HF_TOKEN"))
    
    # Prepare Model and Tokenizer
    model_name = HF_MODEL
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, padding=True, padding_side="left")
    
    if isinstance(model.config.eos_token_id, list):
        tokenizer.pad_token_id = model.config.eos_token_id[0]
    else:
        tokenizer.pad_token_id = model.config.eos_token_id

    df = pd.read_json(DATASET_PATH)[:1]

    logging.info('Labelling Data')
    final_df = run_labelling(df, BATCH_SIZE, model, tokenizer, DEVICE, LLM_MAX_TOKENS, LLM_TEMPERATURE, SAVE_PATH)

    final_df.to_json(os.path.join(SAVE_PATH, 'role_playing_label_outputs.json'))
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-Round Persona Based Claim Generation')
    parser.add_argument('--save_path', type=str, default=os.path.join('script_outputs', 'role_playing_misinformation_labelling_output'), help='Output Location')
    parser.add_argument('--dataset_path', type=str, default=os.path.join('script_outputs', 'role_playing_misinformation_generation_output', 'role_playing_outputs.json'), help='Dataset Location')
    parser.add_argument('--hf_model', type=str, default='meta-llama/Llama-3.1-8B-Instruct', help='HuggingFace Model')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch Size')
    parser.add_argument('--max_tokens', type=float, default=256 * 32, help='LLM Max Tokens')
    parser.add_argument('--temperature', type=float, default=0.7, help='LLM Temperature')
    parser.add_argument('--device', type=str, default='cuda', help='PyTorch Device')
    args = parser.parse_args()
    perform_role_playing_labelling(args)