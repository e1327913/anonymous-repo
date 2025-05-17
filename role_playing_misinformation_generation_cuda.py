import os
import argparse
import logging
import torch
import json
import json_repair
import transformers
import datetime
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import login
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from constants.roles import DEMOCRAT_ROLE, MODERATE_ROLE, REPUBLICAN_ROLE
from utils.pytorch_dataset_classes.misinformation_dataset import MisinformationDataset
from utils.huggingface.functions import step, format_chunks

# Load variables from .env file
load_dotenv()
# Prepare Logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def source_reasoning_prompt(claim_owner, original_claim, sources, role, role_definition):
    prompt = f"""You are a {role}. A {role} is described as follows:
{role_definition}

Your task is to analyze and reason about the original claim, the original claim owner, and its context from your role’s perspective.

Original Claim Owner: {claim_owner}
Original Claim: {original_claim}
Original Claim Context (Sources): {sources}

Your reasoning should address:
1) Role Interpretation:  
   - How does your role perceive this claim? 
   - Does your role agree, disagree, or reframe it differently?  
   - Would your role see this claim as valid, misleading, or needing modification?  

2) Key Emphasis & Framing:  
   - What parts of this claim would your role emphasize or downplay?  
   - How does your role’s beliefs or agenda influence what details are most important?  
 
Summarize your reasoning clearly and concisely before proceeding to modify the claim.  
Ensure that your reasoning remains consistent with the role’s values and agenda.  
"""
    return prompt.strip()

def source_reasoning_with_previous_claims_prompt(claim_owner, original_claim, previous_claims, sources, role, role_definition):
    prompt = f"""You are a {role}. A {role} is described as follows:
{role_definition}

Your task is to analyze and reason about the original claim, the original claim owner, and its context from your role’s perspective.
You should also consider how previous claims have evolved from the original claim and whether your role agrees, reframes, or challenges them.

Original Claim Owner: {claim_owner}
Original Claim: {original_claim}
Original Claim Context (Sources): {sources}
Previous Claims (from previous rounds): {previous_claims}

Your reasoning should address:
1) Role Interpretation:  
   - How does your role perceive these claims?
   - Does your role agree, disagree, or reframe it differently?  
   - Would your role see these claim as valid, misleading, or needing modification?  

2) Key Emphasis & Framing:  
   - What parts of these claim would your role emphasize or downplay?  
   - How does your role’s beliefs or agenda influence what details are most important?  
 
Summarize your reasoning clearly and concisely before proceeding to modify these claims. 
Ensure that your reasoning remains consistent with the role’s values and agenda.  
"""
    return prompt.strip()


def role_playing_claim_prompt():
    prompt = f"""Based on your role’s reasoning, generate a persuasive yet factually coherent claim.

Ensure that:
- The claim remains aligned with the original claim and sources.
- The wording, tone, emphasis, and persuasion reflect your role’s beliefs.
- The claim is natural, engaging, clear, concise, and no more than 20 words.

Modified Claim:
"""
    return prompt.strip()

def intent_claim_prompt():
    prompt = f"""State your intent when generating this claim based on your role.

Consider:
- What message is your role trying to convey with this claim?
- What reaction does your role want to provoke in the audience?
- Does this claim aim to inform, persuade, create doubt, or reinforce a belief?
- How does your role’s ideology shape the claim’s purpose?

Ensure the response is written in a single, coherent sentence.
"""
    return prompt.strip()

def claim_explanation_prompt():
    prompt = f"""Provide a structured explanation of the modified claim.

Your response should include:
- How was the claim modified from the original?
- Why does the modification align with your role’s beliefs and perspective?
- How does the claim remain factually coherent while reflecting your role’s emphasis?
- What effect is the claim intended to have on the audience?

Ensure the explanation flows naturally as a single, concise sentence.
"""
    return prompt.strip()

def format_output_prompt():
    prompt = f"""Return the Claim, Intent, and Explanation in JSON Format.

Ensure that:
- The Claim remains aligned with the original claim and sources.
- The Intent clearly defines the purpose of the claim.
- The Explanation justifies the claim’s modification while maintaining logical consistency.

Format the response as follows:

```json
{{
  "Claim": "<Modified claim>",
  "Intent": "<Purpose of the claim>",
  "Explanation": "<How and why the claim was modified>"
}}
"""
    return prompt.strip()


def collate_fn(batch):
    return batch  # Simply return the list of dictionaries

def format_round_1_df(dataframe, roles):
    role_definitions = [item["role_refinement"] for item in roles]
    roles = [item["name"] for item in roles]
    # Repeat each row 3 times
    df = dataframe.loc[dataframe.index.repeat(3)].reset_index(drop=True)
    
    # Assign roles
    df["role"] = roles * (len(df) // len(roles))
    df["role_definition"] = role_definitions * (len(df) // len(roles))
    df["role_sequence"] = roles * (len(df) // len(roles))
    df["round"] = 0
    df['previous_claims'] = None
    df = df.explode("role", ignore_index=True).explode("role_sequence", ignore_index=True)
    return df

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

def run_round_1(dataframe, batch_size, model, tokenizer, device, max_generated_tokens, temperature):
    # Prepare Base Prompts
    histories = []
    for row in dataframe.iloc():
        claim_owner = row['claim_owner'].strip()
        claim = row['claim'].strip()
        sources = ' '.join(row['misinformation_sources'])
        prompt = source_reasoning_prompt(claim_owner, claim, sources, row['role'], row['role_definition'])
        histories.append([{'role': 'user', 'content': prompt}])
    
    histories = execute_prompts(histories, batch_size, model, tokenizer, device, max_generated_tokens, temperature)

    # Role-Playing Prompt
    for idx, row in enumerate(dataframe.iloc()):
        prompt = role_playing_claim_prompt()
        histories[idx].append({'role': 'user', 'content': prompt})

    histories = execute_prompts(histories, batch_size, model, tokenizer, device, max_generated_tokens, temperature)

    # Intent Prompt
    for idx, row in enumerate(dataframe.iloc()):
        prompt = intent_claim_prompt()
        histories[idx].append({'role': 'user', 'content': prompt})

    histories = execute_prompts(histories, batch_size, model, tokenizer, device, max_generated_tokens, temperature)

    # Explanation Prompt
    for idx, row in enumerate(dataframe.iloc()):
        prompt = claim_explanation_prompt()
        histories[idx].append({'role': 'user', 'content': prompt})

    histories = execute_prompts(histories, batch_size, model, tokenizer, device, max_generated_tokens, temperature)

    # Format Prompt
    for idx, row in enumerate(dataframe.iloc()):
        prompt = format_output_prompt()
        histories[idx].append({'role': 'user', 'content': prompt})

    histories = execute_prompts(histories, batch_size, model, tokenizer, device, max_generated_tokens, temperature)

    # Return all generated claims
    generated_claims = []
    for idx, history in enumerate(histories):
        claim_obj = json_repair.loads(history[-1]['content'])
        claim_obj['round'] = 0
        claim_obj['role'] = dataframe.iloc[idx]['role']
        generated_claims.append(claim_obj)

    dataframe['generated_claims'] = generated_claims
    dataframe['generated_histories'] = histories

    return dataframe

def format_round_x_df(dataframe, roles):
    round_x_data = []
    for _, row in dataframe.iterrows():
        previous_round_roles = row['role_sequence'].split('->')
        next_roles = []
        for role in roles:
            if role['name'] not in previous_round_roles:
                next_roles.append(role)
        
        for next_role in next_roles:
            new_row = row.copy()
            new_row['round'] = new_row['round'] + 1
            new_row["role"] = next_role['name']
            new_row["role_definition"] = next_role['role_refinement']
            new_row["role_sequence"] = f"{row['role_sequence']}->{next_role['name']}"
            if new_row['round'] == 1:
                new_row['previous_claims'] = [new_row[f'generated_claims']]
            else:
                new_row['previous_claims'] = new_row['previous_claims'] + [new_row[f'generated_claims']]

            # logging.info(f'Current Round: {new_row['round']}')
            if new_row['round'] == 1:
                assert len(new_row['previous_claims']) == 1
            elif new_row['round'] == 2:
                assert len(new_row['previous_claims']) == 2

            round_x_data.append(new_row)
    
    df = pd.DataFrame(round_x_data)
    return df

def run_round_x(dataframe, round_idx, batch_size, model, tokenizer, device, max_generated_tokens, temperature, save_path):
    # Prepare Base Prompts
    histories = []
    prompt_path = os.path.join(save_path, f'reason_{round_idx + 1}.json')
    if not os.path.exists(prompt_path):
        logging.info(f"Running Reasoning Prompt Round {round_idx + 1}")
        for row in dataframe.iloc():
            claim_owner = row['claim_owner'].strip()
            claim = row['claim'].strip()
            previous_claims = '.'.join([f"Person with a {item['role']} role claimed that {item['Claim']}" for item in row['previous_claims']])
            sources = ' '.join(row['misinformation_sources'])
            prompt = source_reasoning_with_previous_claims_prompt(claim_owner, claim, previous_claims, sources, row['role'], row['role_definition'])
            histories.append([{'role': 'user', 'content': prompt}])
        
        histories = execute_prompts(histories, batch_size, model, tokenizer, device, max_generated_tokens, temperature)

        with open(prompt_path, 'w') as f:
            json.dump(histories, f)

    rp_path = os.path.join(save_path, f'role_playing_{round_idx + 1}.json')
    if not os.path.exists(rp_path):
        logging.info(f"Running Role-Playing Prompt Round {round_idx + 1}")
        with open(prompt_path, 'r') as f:
            histories = json.load(f)

        # Role-Playing Prompt
        for idx, row in enumerate(dataframe.iloc()):
            prompt = role_playing_claim_prompt()
            histories[idx].append({'role': 'user', 'content': prompt})

        histories = execute_prompts(histories, batch_size, model, tokenizer, device, max_generated_tokens, temperature)
        with open(rp_path, 'w') as f:
            json.dump(histories, f)

    # Intent Prompt
    intent_path = os.path.join(save_path, f'intent_{round_idx + 1}.json')
    if not os.path.exists(intent_path):
        logging.info(f"Running Intent Prompt Round {round_idx + 1}")
        with open(rp_path, 'r') as f:
            histories = json.load(f)

        for idx, row in enumerate(dataframe.iloc()):
            prompt = intent_claim_prompt()
            histories[idx].append({'role': 'user', 'content': prompt})

        histories = execute_prompts(histories, batch_size, model, tokenizer, device, max_generated_tokens, temperature)
        with open(intent_path, 'w') as f:
            json.dump(histories, f)

    # Explanation Prompt
    exp_path = os.path.join(save_path, f"explanation_{round_idx + 1}.json")
    if not os.path.exists(exp_path):
        logging.info(f"Running Explanation Prompt Round {round_idx + 1}")
        with open(intent_path, 'r') as f:
            histories = json.load(f)

        for idx, row in enumerate(dataframe.iloc()):
            prompt = claim_explanation_prompt()
            histories[idx].append({'role': 'user', 'content': prompt})

        histories = execute_prompts(histories, batch_size, model, tokenizer, device, max_generated_tokens, temperature)
        with open(exp_path, 'w') as f:
            json.dump(histories, f)

    # Format Prompt
    format_path = os.path.join(save_path, f"formatted_{round_idx + 1}.json")
    if not os.path.exists(format_path):
        logging.info(f"Running Formatting Prompt Round {round_idx + 1}")
        with open(exp_path, 'r') as f:
            histories = json.load(f)

        for idx, row in enumerate(dataframe.iloc()):
            prompt = format_output_prompt()
            histories[idx].append({'role': 'user', 'content': prompt})

        histories = execute_prompts(histories, batch_size, model, tokenizer, device, max_generated_tokens, temperature)

        with open(format_path, 'w') as f:
            json.dump(histories, f)

    if os.path.exists(format_path):
        logging.info(f"Running Final Formatting Round {round_idx + 1}")
        with open(format_path, 'r') as f:
            histories = json.load(f)
        # Return all generated claims
        generated_claims = []
        for idx, history in enumerate(histories):
            claim_obj = json_repair.loads(history[-1]['content'])
            claim_obj['round'] = round_idx
            claim_obj['role'] = dataframe.iloc[idx]['role']
            generated_claims.append(claim_obj)

        dataframe['generated_claims'] = generated_claims
        dataframe['generated_histories'] = histories

    return dataframe

def perform_role_playing_generation(args):
    # Argparse
    DEVICE = args.device
    HF_MODEL = args.hf_model
    SAVE_PATH = args.save_path
    DATASET_PATH = args.dataset_path
    BATCH_SIZE = args.batch_size
    LLM_MAX_TOKENS = args.max_tokens
    LLM_TEMPERATURE = args.temperature
    MISINFORMATION_ROLES = [DEMOCRAT_ROLE, REPUBLICAN_ROLE, MODERATE_ROLE]

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

    df = pd.read_json(DATASET_PATH)

    if not os.path.exists(os.path.join(SAVE_PATH, 'role_playing_outputs_round_1.json')):
        # Prepare Round 1 Dataset
        logging.info('Formatting Round 1')
        df = format_round_1_df(df, MISINFORMATION_ROLES)

        # Run Round 1
        logging.info('Run Round 1')
        round_1_df = run_round_1(df, BATCH_SIZE, model, tokenizer, DEVICE, LLM_MAX_TOKENS, LLM_TEMPERATURE)
        
        round_1_df.to_json(os.path.join(SAVE_PATH, 'role_playing_outputs_round_1.json'), index=False)

    if not os.path.exists(os.path.join(SAVE_PATH, 'role_playing_outputs_round_2.json')):
        # Prepare Round 2 Dataset
        logging.info('Formatting Round 2')
        round_1_df = pd.read_json(os.path.join(SAVE_PATH, 'role_playing_outputs_round_1.json'))
        df = format_round_x_df(round_1_df, MISINFORMATION_ROLES)
        
        # Run Round 2
        logging.info('Run Round 2')
        round_2_df = run_round_x(df, 1, BATCH_SIZE, model, tokenizer, DEVICE, LLM_MAX_TOKENS, LLM_TEMPERATURE, SAVE_PATH)
        round_1_and_2_final_df = pd.concat([round_1_df, round_2_df], ignore_index=True)
        round_1_and_2_final_df.to_json(os.path.join(SAVE_PATH, 'role_playing_outputs_round_2.json'))

    if not os.path.exists(os.path.join(SAVE_PATH, 'role_playing_outputs.json')):
        # Prepare Round 3 Dataset
        logging.info('Formatting Round 3')
        round_1_df = pd.read_json(os.path.join(SAVE_PATH, 'role_playing_outputs_round_1.json'))
        round_1_and_2_df = pd.read_json(os.path.join(SAVE_PATH, 'role_playing_outputs_round_2.json'))
        round_2_df = round_1_and_2_df[round_1_and_2_df['round'] == 1].reset_index(drop=True)
        df = format_round_x_df(round_2_df, MISINFORMATION_ROLES)
        assert len(df) == len(round_2_df)

        # Run Round 3
        logging.info('Run Round 3')
        round_3_df = run_round_x(df, 2, BATCH_SIZE, model, tokenizer, DEVICE, LLM_MAX_TOKENS, LLM_TEMPERATURE, SAVE_PATH)
        final_df = pd.concat([round_1_df, round_2_df, round_3_df], ignore_index=True)
        final_df.to_json(os.path.join(SAVE_PATH, 'role_playing_outputs.json'))
    
    
if __name__ == '__main__':
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    parser = argparse.ArgumentParser(description='Multi-Round Persona Based Claim Generation')
    parser.add_argument('--save_path', type=str, default=os.path.join('script_outputs', 'role_playing_misinformation_generation_output'), help='Output Location')
    parser.add_argument('--dataset_path', type=str, default=os.path.join('script_outputs', 'train_dev_test_split', 'test.json'), help='Dataset Location')
    parser.add_argument('--hf_model', type=str, default='Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2', help='HuggingFace Model')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch Size')
    parser.add_argument('--max_tokens', type=float, default=256 * 32, help='LLM Max Tokens')
    parser.add_argument('--temperature', type=float, default=0.7, help='LLM Temperature')
    parser.add_argument('--device', type=str, default='cuda', help='PyTorch Device')
    args = parser.parse_args()
    perform_role_playing_generation(args)