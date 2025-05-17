import os
import argparse
import logging
import json
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken

# Load variables from .env file
load_dotenv()
# Prepare Logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def labelling_prompt(claim, evidence):
    return f"""You are a fact-checking assistant. Your task is to analyze the claim and compare it with the provided evidence.

Claim:  
"{claim}"

Evidence:  
"{evidence}"

Instructions:
1. Carefully analyze whether the claim is fully, partially, or not supported by the evidence.
2. Identify specific factual elements in the claim that are supported or contradicted.
3. Note any missing context, exaggerations, or misleading aspects of the claim.
4. Do not make assumptions beyond what the evidence explicitly states.

=== 

Based on your factual analysis, assign the appropriate label.

Label Definitions:
- True: A statement is fully accurate.
- Half-True: A statement that conveys only part of the truth, especially one used deliberately in order to mislead someone.
- False: A statement is inaccurate or contradicted by evidence.

Instructions:
- Avoid assumptions beyond what the analysis states.
- Ensure consistency between the label and reasoning.
- Provide your confidence score for all of the labels.

===
Select the label based on the highest confidence score and provide an explanation on your factual analysis.

Output Format (JSON)
```json
{{
  "Label": "<True / Half-True / False>",
  "Explanation": "<Short justification referencing your factual analysis>"
}}
"""

def perform_role_playing_labelling(args):
    # Argparse
    SAVE_PATH = args.save_path
    DATASET_PATH = args.dataset_path

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    # Save current args to output
    with open(f"{SAVE_PATH}/argsparse_config.json", 'w') as file:
        json.dump(vars(args), file)
        file.close()

    df = pd.read_json(DATASET_PATH)
    client = OpenAI(api_key=os.getenv("OPENAI_KEY"))
    prompts = []

    for _, row in df.iterrows():
        prompts.append({
            'url': f"{round}_{row['role']}_{row['url']}_{row['role_sequence']}", 
            "prompt": labelling_prompt(row['generated_claims']['Claim'], ' '.join(row['fact_checking_evidences']))
        })

    encoder = tiktoken.encoding_for_model('gpt-4o-mini')
    request_file_path = os.path.join(SAVE_PATH, f'labelling_prompt.jsonl')

    with open(request_file_path, 'w') as outfile:
        for prompt in prompts:
            data = {
                "custom_id": prompt['url'],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "user", "content": prompt['prompt']},
                    ],
                }
            }
            json.dump(data, outfile)
            outfile.write('\n')
        outfile.close()
    
    batch_input_file = client.files.create(file=open(request_file_path, "rb"), purpose="batch")
    batch_input_file_id = batch_input_file.id
    output = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": f"Factuality Check for With and Without Role Definitions claims"
        }
    )

    with open(os.path.join(SAVE_PATH, 'output.json'), 'w') as outfile:
        detail = {
            "batch_id": output.id,
            "original_file": request_file_path
        }
        json.dump(detail, outfile)
        outfile.close()

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-Round Persona Based Claim Generation')
    parser.add_argument('--save_path', type=str, default=os.path.join('ablation_script_outputs', 'labelling'), help='Output Location')
    parser.add_argument('--dataset_path', type=str, default=os.path.join('script_outputs', 'role_playing_misinformation_generation_output', '2025-03-21-00-48-13', 'role_playing_outputs.json'), help='Dataset Location')
    args = parser.parse_args()
    perform_role_playing_labelling(args)