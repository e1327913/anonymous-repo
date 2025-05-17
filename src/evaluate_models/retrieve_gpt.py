import os
import argparse
import json
import json_repair
import pandas as pd
from openai import OpenAI
from tqdm.auto import tqdm
# importing necessary functions from dotenv library
from dotenv import load_dotenv
# loading variables from .env file
load_dotenv()

def start_evaluation(args):
    # Argparse
    SAVE_PATH = args.save_path
    ORIGINAL_DATA_PATH = args.original_data_path
    
    client = OpenAI(api_key=os.getenv("OPENAI_KEY"))
    
    openai_df = pd.DataFrame([])
    for prompt_type in os.listdir(SAVE_PATH):
        prompt_path = os.path.join(SAVE_PATH, prompt_type)
        if not os.path.isdir(prompt_path):
            continue
        for dataset_type in os.listdir(prompt_path):
            if dataset_type == '.DS_Store':
                continue
            data_path = os.path.join(prompt_path, dataset_type, 'output.json')
            prompt_data_path = os.path.join(prompt_path, dataset_type, 'eval_requests_0.jsonl')
            original_data = os.path.join(ORIGINAL_DATA_PATH, f"{dataset_type}.json")
            batch_df = pd.read_json(data_path, lines=True)
            prompt_df = pd.read_json(prompt_data_path, lines=True)
            test_df = pd.read_json(original_data)
            batch_details = client.batches.retrieve(batch_df['batch_id'].iloc()[0])
            if batch_details.status == 'completed':
                file_response = client.files.content(batch_details.output_file_id)
                json_output = json_repair.loads(file_response.text)
                labels, explanations, prompts, original_labels = [], [], [], []
                # Extract output from OpenAI
                for item in tqdm(json_output):
                    current_response = item['response']['body']['choices']
                    custom_id = item['custom_id'].split('@')
                    assert len(custom_id) == 2 and len(current_response) == 1
                    current_response = json_repair.loads(current_response[0]['message']['content'])
                    prompt_row = prompt_df[prompt_df['custom_id'] == item['custom_id']].iloc[0]
                    prompt = prompt_row['body']['messages'][0]['content']
                    original_label = test_df[test_df['id'] == int(custom_id[1])].iloc[0]['label'].lower()
                    assert len(original_label) > 1
                    original_labels.append(original_label)
                    labels.append(current_response['Label'].lower())
                    explanations.append(current_response['Explanation'] if 'Explanation' in current_response else None)
                    prompts.append(prompt)
                
                curr_df = pd.DataFrame({
                    'model': ['gpt-4o-mini' for _ in range(len(prompts))],
                    'dataset_type': [dataset_type for _ in range(len(prompts))],
                    'prompt_type': [prompt_type for _ in range(len(prompts))],
                    'pred_label': labels,
                    'pred_explanation': explanations,
                    'pred_prompts': prompts,
                    'true_label': original_labels
                })
                openai_df = pd.concat([openai_df, curr_df])

            else:
                print(f"Original Details: {batch_details}")
                print(f"Status: {batch_details.status}")
                print(f"Reason: {batch_details.errors}")
                
    openai_df = openai_df.reset_index(drop=True)
    openai_df.to_json(os.path.join(SAVE_PATH, 'gpt_evaluation.json'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Decoder Evaluation')
    parser.add_argument("--save_path", type=str, default=os.path.join('script_outputs', 'decoder_models_results', 'gpt-4o-mini'))
    parser.add_argument("--original_data_path", type=str, default=os.path.join('script_outputs', 'role_playing_misinformation_generated_datasets', '2025-03-21-00-48-13'))
    args = parser.parse_args()
    start_evaluation(args)