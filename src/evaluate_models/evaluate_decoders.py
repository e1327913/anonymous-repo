import os
import argparse
import logging
import torch
import json_repair
import json
import transformers
import pandas as pd
from prompts.evaluation_prompts import zero_shot_evaluation_prompt, zero_shot_evaluation_prompt_cot, few_shot_evaluation_prompt, few_shot_evaluation_prompt_cot
from dotenv import load_dotenv
from huggingface_hub import login
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# Load variables from .env file
load_dotenv()
# Prepare Logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class PromptDataset(Dataset):
    def __init__(self, prompts):
        self.prompts = prompts
    
    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index):
        return self.prompts[index]

def step(model, tokenizer, histories, device, max_tokens=256 * 16):
    # Tokenize Prompts
    prompts = tokenizer.apply_chat_template(conversation=histories, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompts, padding=True, return_tensors="pt")
    inputs = inputs.to(device)
    # Text Generation
    outputs = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True)
    # Decode Outputs
    outputs = [tokenizer.decode(output) for output in outputs]
    return outputs

def format_model_chunks(item):
    new_history = []
    chunks = item.replace('<|begin_of_text|>', '').replace('<|end_of_text|>', '').split('<|eot_id|>')
    for chunk in chunks:
        details = [detail for detail in chunk.split('\n\n')]
        role = details[0].replace('<|start_header_id|>', '').replace('<|end_header_id|>', '').strip()
        content = ' '.join(details[1:]).strip()
        content = json_repair.loads(content)
        if len(role) > 0 and len(content) > 0:
            new_history.append({'role': role, 'content': content})
    return new_history

def is_output_valid(llm_dict, keys=['Label', 'Explanation']):
    # Make sure that the LLM output is a Dictionary
    if not isinstance(llm_dict, dict):
        return False
    
    for key in keys:
        if key not in llm_dict.keys():
            return False
    
    for key in llm_dict:
        if isinstance(llm_dict[key], str) and len(llm_dict[key]) < 2:
            return False
        
    return True

def perform_evaluation(model, tokenizer, histories, device, max_tokens=256*16):
    outputs = step(model, tokenizer, histories, device, max_tokens)
    # Format & Validate LLM Outputs
    formatted_outputs = []
    for index, output in enumerate(outputs):
        new_history = format_model_chunks(output)
        new_content = new_history[-1]['content']
        logging.info(new_content)
        is_content_valid = is_output_valid(new_content)
        logging.info(f"Is Content Valid: {is_content_valid}")
        if is_content_valid:
            formatted_outputs.append(new_content)
            histories[index].append(new_history[-1])
        else:
            is_not_valid = True
            while is_not_valid:
                logging.info(f"Retrying at History {index}")
                specified_hf_outputs = step(model, tokenizer, histories[index], device, max_tokens)
                assert len(specified_hf_outputs) == 1
                specified_hf_output = specified_hf_outputs[0]
                specified_new_history = format_model_chunks(specified_hf_output)
                specified_new_content = specified_new_history[-1]['content']
                logging.info(specified_new_content)
                is_content_valid = is_output_valid(specified_new_content)
                logging.info(f"Is Content Valid: {is_content_valid}")
                if is_content_valid:
                    formatted_outputs.append(specified_new_content)
                    histories[index].append(specified_new_history[-1])
                    is_not_valid = False
    return formatted_outputs, histories
    
def prepare_evaluation_prompts(dataframe, mode='zero_shot'):
    prompts = []
    for row in dataframe.iloc():
        claim = row['claim']
        fact_checking_evidences = row['fact_checking_evidence']
        fce_dict = {}
        for index, text in enumerate(fact_checking_evidences):
            fce_dict[f"fce_source_{index}"] = text
        prompt = None
        if mode == 'zero_shot':
            prompt = zero_shot_evaluation_prompt(claim, fce_dict)
        elif mode == 'few_shot':
            prompt = few_shot_evaluation_prompt(claim, fce_dict)
        elif mode == 'zero_shot_cot':
            prompt = zero_shot_evaluation_prompt_cot(claim, fce_dict)
        elif mode == 'few_shot_cot':
            prompt = few_shot_evaluation_prompt_cot(claim, fce_dict)
        else:
            raise ValueError(mode)
        prompts.append(prompt)
    return prompts
    
def start_evaluation(args):
    # Argparse
    HF_MODEL = args.hf_model
    MODE = args.mode
    SAVE_PATH = args.save_path
    EVAL_DATA_PATH = args.eval_data_path
    EVAL_BATCH_SIZE = args.eval_batch_size
    LLM_MAX_TOKENS = args.max_tokens
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    # Save current args to output
    with open(f"{SAVE_PATH}/argsparse_config.json", 'w') as file:
        json.dump(vars(args), file)
        file.close()

    # Login to HF 
    # login(os.getenv("HF_TOKEN"))
    login("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")
    
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
        
    dataframe = pd.read_json(EVAL_DATA_PATH)
    prompts = prepare_evaluation_prompts(dataframe, MODE)
    dataset = PromptDataset(prompts)
    dataloader = DataLoader(dataset, EVAL_BATCH_SIZE)
    
    details = {
        'pred_label': [],
        'pred_label_explanation': [],
        'pred_label_histories': [],
        'pred_label_prompt': []
    }
    for batch_prompts in tqdm(dataloader):
        batch_histories = [[{'role': 'user', 'content': prompt}] for prompt in batch_prompts]
        batch_outputs, batch_histories = perform_evaluation(model, tokenizer, batch_histories, DEVICE, LLM_MAX_TOKENS)
        details['pred_label'] += [item['Label'] for item in batch_outputs]
        details['pred_label_explanation'] += [item['Explanation'] for item in batch_outputs]
        details['pred_label_histories'] += batch_histories
        details['pred_label_prompt'] += batch_prompts
    
    dataframe['pred_label'] = details['pred_label']
    dataframe['pred_label_explanation'] = details['pred_label_explanation']
    dataframe['pred_label_histories'] = details['pred_label_histories']
    dataframe['pred_label_prompt'] = details['pred_label_prompt']
    dataframe['mode'] = [MODE for _ in range(len(dataframe))]
    dataframe.to_json(os.path.join(SAVE_PATH, 'eval.json'))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Decoder Evaluation')
    
    # Data Arguments
    parser.add_argument("--save_path", type=str, default=os.path.join('script_outputs', 'decoder_models_results', 'llama-3.1-8B-Instruct'))
    parser.add_argument("--eval_data_path", type=str, default=os.path.join('script_outputs', 'role_playing_misinformation_generated_datasets', '2025-03-21-00-48-13', 'mixed.json'))
    parser.add_argument('--hf_model', type=str, default='meta-llama/Llama-3.1-8B-Instruct', help='Model')
    parser.add_argument('--max_tokens', type=float, default=256 * 32, help='Script output location')
    parser.add_argument('--mode', type=str, default='zero_shot', help='Script output location')
    
    # Eval Arguments
    parser.add_argument("--eval_batch_size", type=int, default=8)
    
    args = parser.parse_args()
    start_evaluation(args)