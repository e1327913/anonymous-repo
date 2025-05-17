import os
import argparse
import datetime
import pandas as pd
import random
import json
import unicodedata
import uuid

def generate_questionnaire_data(args):
    # Argparse
    RANDOM_SEED = args.random_seed
    SAVE_PATH = args.save_path
    DATASET_PATH = args.dataset_path
    
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    # Save current args to output
    with open(f"{SAVE_PATH}/argsparse_run_config.json", 'w') as file:
        json.dump(vars(args), file)
        file.close()

    df = pd.DataFrame([])
    paths = [path for path in os.listdir(DATASET_PATH) if 'argsparse' not in path]
    for data in paths:
        data_df = pd.read_json(os.path.join(DATASET_PATH, data))
        df = pd.concat([df, data_df])

    # For 450 Claims, we will sample 30 URLs
    unique_urls = df["url"].drop_duplicates()
    sampled_urls = unique_urls.sample(n=30, random_state=RANDOM_SEED).tolist()

    sampled_df = df[df['url'].isin(sampled_urls)]
    groups = sampled_df.groupby('url')

    sampled_data = {
        'id': [],
        'url': [],
        'round': [],
        'role_sequence': [],
        'generated_claim': [],
        'generated_intent': [],
        'sources': [],
        'previous_claims': [],
        'evidences': [],
        'role': [],
        'role_description': [],
    }
    for url, group in groups:
        for row in group.iloc():
            sampled_data['id'].append(str(uuid.uuid4()))
            sampled_data['url'].append(url)
            sampled_data['round'].append(row['round'])
            sampled_data['role_sequence'].append(row['role_sequence'])
            sampled_data['generated_claim'].append(unicodedata.normalize("NFKD", row['generated_claims']['Claim'].strip()))
            sampled_data['generated_intent'].append(row['generated_claims']['Intent'].strip())
            sampled_data['sources'].append(unicodedata.normalize("NFKD", '@'.join(row['misinformation_sources'])))
            sampled_data['evidences'].append(unicodedata.normalize("NFKD", '@'.join(row['fact_checking_evidences'])))
            sampled_data['role'].append(row['role'])
            sampled_data['role_description'].append(row['role_definition'])

            if row['round'] == 0:
                sampled_data['previous_claims'].append(None)
            elif row['round'] == 1:
                sampled_data['previous_claims'].append(f"{row['previous_claims'][0]['role']}: {row['previous_claims'][0]['Claim']}")
            elif row['round'] == 2:
                sampled_data['previous_claims'].append('@'.join([f"{item['role']}: {item['Claim']}" for item in row['previous_claims']]))

    sampled_df = pd.DataFrame(sampled_data).sort_values(['url', 'round'])
    sampled_df.to_csv(os.path.join(SAVE_PATH, 'questionnaire_data.csv'))

if __name__ == '__main__':
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    parser = argparse.ArgumentParser(description='Politifact Link Extractor. Extract Personalities and Article Links from targeted URLs')
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--save_path', type=str, default=os.path.join('script_outputs', 'questionnaire'), help='Script output location')
    parser.add_argument('--dataset_path', type=str, default=os.path.join('script_outputs', 'role_playing_misinformation_labelling_output', '2025-03-21-00-48-13'))
    args = parser.parse_args()
    generate_questionnaire_data(args)
