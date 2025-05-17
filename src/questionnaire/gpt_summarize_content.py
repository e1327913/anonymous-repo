import os
import argparse
import logging
import datetime
import json
import pandas as pd
import tiktoken
import json_repair
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# Load variables from .env file
load_dotenv()
# Prepare Logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def role_summarize_prompt(role, role_definition):
    return f"""Summarize the following sources into 3 or fewer bullet points that are short, clear, easy to read and remains consistent to the original definition.
Role: {role}
Role Definition: 
{role_definition}
    """

def sources_summarize_prompt(sources):
    sources = '\n'.join([f"- {text}" for text in sources.split('@')])
    return f"""Summarize the following sources into 3 or fewer bullet points that are short, clear, easy to read and remains consistent to the original sources.
Sources:
{sources}
    """

def evidence_summarize_prompt(evidence):
    evidence = '\n'.join([f"- {text}" for text in evidence.split('@')])
    return f"""Summarize the following sources into 3 or fewer bullet points that are short, clear, easy to read and remains consistent to the original evidence.
Evidence:
{evidence}
    """

def gpt_summarize_content(args):
    # Argparse
    SAVE_PATH = args.save_path
    DATASET_PATH = args.dataset_path
    
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    # Save current args to output
    with open(f"{SAVE_PATH}/argsparse_run_config.json", 'w') as file:
        json.dump(vars(args), file)
        file.close()
        
    df = pd.read_csv(DATASET_PATH)
    client = OpenAI(api_key=os.getenv("OPENAI_KEY"))
    
    prompts = {
        'role': [],
        'sources': [],
        'evidences': []
    }

    # Role Prompt
    for row in df[['role', 'role_description']].drop_duplicates().iloc():
        prompt = role_summarize_prompt(row['role'], row['role_description'])
        prompts['role'].append({'unique_key': row['role'], 'prompt': prompt})
    
    # Misinformation Sources
    for row in df[['url', 'sources']].drop_duplicates().iloc():
        prompt = sources_summarize_prompt(row['sources'])
        prompts['sources'].append({'unique_key': f"{row['url']}<|split|>source", 'prompt': prompt})

    # Fact-Checking Evidence
    for row in df[['url', 'evidences']].drop_duplicates().iloc():
        prompt = evidence_summarize_prompt(row['evidences'])
        prompts['evidences'].append({'unique_key': f"{row['url']}<|split|>evidence", 'prompt': prompt})

    output_file_names = []
    encoder = tiktoken.encoding_for_model('gpt-4o-mini')
    for key in prompts:
        texts = 0
        current_prompts = prompts[key]
        request_file_path = os.path.join(SAVE_PATH, f'questionnaire_summarize_{key}.jsonl')
        with open(request_file_path, 'w') as outfile:
            for prompt_data in current_prompts:
                data = {
                    "custom_id": prompt_data['unique_key'],
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-4o-mini",
                        "messages": [
                            {"role": "user", "content": prompt_data['prompt']},
                        ],
                    }
                }
                tokens = encoder.encode(prompt_data['prompt'])
                texts += len(tokens)
                json.dump(data, outfile)
                outfile.write('\n')
            outfile.close()
        print(texts)
        output_file_names.append(request_file_path)
    
    final_details = []
    for file_path in tqdm(output_file_names):
        batch_input_file = client.files.create(
            file=open(file_path, "rb"),
            purpose="batch"
        )
        batch_input_file_id = batch_input_file.id
        output = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": f"Politifact Misinformation Sources & Fact-Checking Extraction with file {file_path}"
            }
        )
        print(output)
        final_details.append({
            "batch_id": output.id,
            "original_file": file_path
        })

    with open(os.path.join(SAVE_PATH, 'output.json'), 'w') as outfile:
        for detail in tqdm(final_details):
            json.dump(detail, outfile)
            outfile.write('\n')
        outfile.close()
    
if __name__ == '__main__':
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    parser = argparse.ArgumentParser(description='Politifact Link Extractor. Extract Personalities and Article Links from targeted URLs')
    parser.add_argument('--save_path', type=str, default=os.path.join('script_outputs', 'questionnaire', 'summarized_content'), help='Script output location')
    parser.add_argument('--dataset_path', type=str, default=os.path.join('script_outputs', 'questionnaire', 'questionnaire_data.csv'))
    args = parser.parse_args()
    gpt_summarize_content(args)