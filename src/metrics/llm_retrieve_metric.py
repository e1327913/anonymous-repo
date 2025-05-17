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
    data_df = pd.read_json(DATASET_PATH)
    
    details = []
    
    with open(os.path.join(SAVE_PATH, 'openai_response.jsonl'), 'w') as outfile:
        for row in tqdm(openai_df.iloc()):
            batch_details = client.batches.retrieve(row['batch_id'])
            if batch_details.status == 'completed':
                file_response = client.files.content(batch_details.output_file_id)
                json_output = json_repair.loads(file_response.text)
                formatted_output = {}
                for idx, item in enumerate(json_output):
                    current_response = item['response']['body']['choices']
                    custom_id = item['custom_id'].split('_')
                    assert len(custom_id) == 5 and len(current_response) == 1
                    role_sequence = custom_id[4]
                    url = custom_id[3]
                    round = custom_id[2]
                    role = custom_id[1]
                    question_key = custom_id[0]
                    current_response = json_repair.loads(current_response[0]['message']['content'])
                    original_details = data_df.loc[(data_df['url'] == url) & (data_df['role'] == role) & (data_df['role_sequence'] == role_sequence) & (data_df['round'] == int(round)), ['generated_claims']].values.tolist()
                    assert len(original_details) == 1
                    original_details = original_details[0]
                    claim_details = original_details[0]
                    formatted_output['url'] = url
                    formatted_output['role'] = role
                    formatted_output['round'] = round
                    formatted_output['role_sequence'] = role_sequence
                    formatted_output['claim'] = claim_details['Claim']
                    formatted_output['intent'] = claim_details['Intent']
                    formatted_output['label'] = None

                    formatted_output['question_key'] = question_key
                    formatted_output['answer'] = current_response['Answer']
                    formatted_output['explanation'] = current_response['Explanation']
                    details.append(formatted_output)
                    formatted_output = {}
                
                assert len(details) > 0
                result_df = pd.DataFrame(details)
                result_df.to_json(os.path.join(SAVE_PATH, 'dataframe.json'))
                json.dump(file_response.text, outfile)
                outfile.write('\n')
            else:
                print(f"Status: {batch_details.status}")
                print(f"Reason: {batch_details.errors}")
        outfile.close()
    
if __name__ == '__main__':
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    parser = argparse.ArgumentParser(description='Politifact Link Extractor. Extract Personalities and Article Links from targeted URLs')
    parser.add_argument('--save_path', type=str, default=os.path.join('script_outputs', 'gpt_4o_mini_metric_request'), help='Script output location')
    parser.add_argument('--dataset_path', type=str, default=os.path.join('script_outputs', 'role_playing_misinformation_labelling_output', 'role_playing_label_outputs.json'))
    args = parser.parse_args()
    retrieve_gpt_metrics(args)