import os
import argparse
import logging
import datetime
import json
import pandas as pd
import json_repair
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# Load variables from .env file
load_dotenv()
# Prepare Logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def retrieve_gpt_metrics(args):
    # Argparse
    SAVE_PATH = args.save_path
    DATASET_PATH = args.dataset_path
    
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    # Save current args to output
    with open(f"{SAVE_PATH}/argsparse_config.json", 'w') as file:
        json.dump(vars(args), file)
        file.close()
        
    # OpenAI Client
    client = OpenAI(api_key=os.getenv("OPENAI_KEY"))
    request_file_path = os.path.join(SAVE_PATH, 'output.json')
    openai_df = pd.read_json(path_or_buf=request_file_path, lines=True)
    data_df = pd.read_csv(DATASET_PATH)
    
    roles = {}
    source_url_content = {}
    evidence_url_content = {}
    with open(os.path.join(SAVE_PATH, 'openai_response.jsonl'), 'w') as outfile:
        for row in tqdm(openai_df.iloc()):
            batch_details = client.batches.retrieve(row['batch_id'])
            if batch_details.status == 'completed':
                file_response = client.files.content(batch_details.output_file_id)
                json_output = json_repair.loads(file_response.text)
                for idx, item in enumerate(json_output):
                    current_response = item['response']['body']['choices']
                    custom_id = item['custom_id'].split('<|split|>')
                    assert len(custom_id) < 3 and len(current_response) == 1
                    if custom_id[0] in ['Democrat', 'Republican', 'Moderate']:
                        roles[custom_id[0]] = current_response[0]['message']['content']
                    elif custom_id[1] == 'source':
                        source_url_content[custom_id[0]] = current_response[0]['message']['content']
                    elif custom_id[1] == 'evidence':
                        evidence_url_content[custom_id[0]] = current_response[0]['message']['content']
                json.dump(file_response.text, outfile)
                outfile.write('\n')
            else:
                print(f"Status: {batch_details.status}")
                print(f"Reason: {batch_details.errors}")
        outfile.close()

    role_def, sources, evidences = [], [], []
    for row in data_df.iloc():
        role_def.append(roles[row['role']])
        sources.append(source_url_content[row['url']])
        evidences.append(evidence_url_content[row['url']])

    data_df['summarized_role_definition'] = role_def
    data_df['summarized_sources'] = sources
    data_df['summarized_evidence'] = evidences
    data_df['assigned_to'] = None

    data_df.to_csv(os.path.join(SAVE_PATH, 'summarized_questionnaire_data.csv'), index=False)
    
if __name__ == '__main__':
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    parser = argparse.ArgumentParser(description='Politifact Link Extractor. Extract Personalities and Article Links from targeted URLs')
    parser.add_argument('--save_path', type=str, default=os.path.join('script_outputs', 'questionnaire', 'summarized_content'), help='Script output location')
    parser.add_argument('--dataset_path', type=str, default=os.path.join('script_outputs', 'questionnaire', 'questionnaire_data.csv'))
    args = parser.parse_args()
    retrieve_gpt_metrics(args)