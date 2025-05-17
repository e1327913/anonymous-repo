import os
import argparse
import logging
import datetime
import json
import pandas as pd
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

# Load variables from .env file
load_dotenv()
# Prepare Logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def relevancy_check(claim, sources, previous_claims):
    return f"""Your task is to answer this question:
    How relevant is the Claim compared to the provided sources and previous claims?
    
    Details:
    - Claim: {claim}
    - Sources: 
        {sources}
    - Previous Claims: 
        {previous_claims}

    Provide a rating based on the following Likert scale:
    - 5: Perfectly Relevant: The claim fully integrates key facts from sources and previous claims.
    - 4: Mostly Relevant: The claim follows sources or previous claims but misses small details.
    - 3: Somewhat Relevant: The claim mentions sources or previous claims but misinterprets or lacks connections.
    - 2: Weakly Relevant: The claim has a weak or indirect connection to the sources or previous claims.
    - 1: Completely Irrelevant: The claim does not relate to any sources or previous claims.

    Response Format:
    {{
        "Question": "Q2",
        "Answer": X,  # (1 to 5)
        "Explanation": "Y"
    }}
    """

def perform_gpt_metrics(args):
    # Argparse
    SAVE_PATH = args.save_path
    BASELINE_DF = args.baseline_df
    TEST_DF = args.test_df
    
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    # Save current args to output
    with open(f"{SAVE_PATH}/argsparse_run_config.json", 'w') as file:
        json.dump(vars(args), file)
        file.close()
        
    df = pd.read_json(TEST_DF)
    baseline_df = pd.read_json(BASELINE_DF)
    urls = list(set(df['url']))
    baseline_df = baseline_df[baseline_df['url'].isin(urls)]
    client = OpenAI(api_key=os.getenv("OPENAI_KEY"))
    prompts = []

    for _, row in df.iterrows():
        baseline_row = baseline_df[(baseline_df['url'] == row['url']) & (baseline_df['role_sequence'] == row['role_sequence']) & (baseline_df['round'] == row['round'])]
        round = row['round']
        sources = '\n'.join([f'- {text}' for text in row['misinformation_sources']])
        previous_claims = None if row['previous_claims'] == None else '\n'.join([f'- Person with a {item['role']} role claimed that {item['Claim']}' for item in row['previous_claims']])
        prompts.append({'url': f"special_{round}_{row['role']}_{row['url']}_{row['role_sequence']}", "prompt": relevancy_check(row['generated_claims']['Claim'], sources, previous_claims)})
        prompts.append({'url': f"baseline_{round}_{row['role']}_{row['url']}_{row['role_sequence']}", "prompt": relevancy_check(baseline_row['generated_claims'].iloc[0]['Claim'], sources, previous_claims)})

    encoder = tiktoken.encoding_for_model('gpt-4o-mini')
    request_file_path = os.path.join(SAVE_PATH, f'relevancy_check.jsonl')

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
            "description": f"Factuality Check for With and Without Background Source claims"
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
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    parser = argparse.ArgumentParser(description='Politifact Link Extractor. Extract Personalities and Article Links from targeted URLs')
    parser.add_argument('--save_path', type=str, default=os.path.join('ablation_script_outputs', 'gpt_4o_mini_factuality_eval'), help='Script output location')
    parser.add_argument('--baseline_df', type=str, default=os.path.join('script_outputs', 'role_playing_misinformation_labelling_output', '2025-03-21-00-48-13', 'role_playing_label_outputs.json'))
    parser.add_argument('--test_df', type=str, default=os.path.join('ablation_script_outputs', 'role_playing_misinformation_generation_output_without_background_sources', '2025-03-21-00-48-13', 'role_playing_outputs.json'))
    args = parser.parse_args()
    perform_gpt_metrics(args)