import os
import argparse
import pandas as pd
import json
import logging
import json_repair
from tqdm import tqdm

# Prepare Logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def format_generated_dataset(dataframe, minimum_count, random_seed):
    data = {
        'id': [],
        'claim': [],
        'fact_checking_evidence': [],
        'label': [],
    }
    
    for idx, row in tqdm(dataframe.iterrows()):
        data['id'].append(idx)
        data['claim'].append(row['generated_claims']['Claim'])
        data['fact_checking_evidence'].append(row['fact_checking_evidences'])
        data['label'].append(row['generated_labels'])
        
    df = pd.DataFrame(data)
    
    # Balance the dataset out
    df = df.groupby('label').sample(n=minimum_count, random_state=random_seed)
    
    logging.info(df['label'].value_counts())
    assert len(set(df['label'])) == 3
    return df

def format_original_dataset(dataframe, minimum_count, random_seed):
    data = {
        'id': [],
        'claim': [],
        'fact_checking_evidence': [],
        'label': [],
    }
    
    for idx, row in tqdm(dataframe.iterrows()):
        data['id'].append(idx)
        data['claim'].append(row['claim'].strip())
        data['fact_checking_evidence'].append(row['fact_checking_evidences'])
        data['label'].append(row['label'])
        
    df = pd.DataFrame(data)
    
    # Balance the dataset out
    df = df.groupby('label').sample(n=minimum_count, random_state=random_seed)
    
    logging.info(df['label'].value_counts())
    assert len(set(df['label'])) == 3
    return df

def generate_test_dataset(args):
    SAVE_PATH = args.save_path
    GENERATED_DATA_PATH = args.generated_data_path
    ORIGINAL_DATA_PATH = args.original_data_path
    RANDOM_SEED = args.random_seed
    
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
        
    # Save current args to output
    with open(f"{SAVE_PATH}/argsparse_config.json", 'w') as file:
        json.dump(vars(args), file)
        file.close() 
    
    df = pd.read_json(GENERATED_DATA_PATH)
    original_df = pd.concat([
        pd.read_json(os.path.join(ORIGINAL_DATA_PATH, 'test.json'))
    ])
    
    # Transform Dataset
    df['generated_labels'] = [json_repair.loads(row['label_histories'][-1]['content'])['Label'].lower() for _, row in df.iterrows()]
    df = df[df['generated_labels'].isin(['true', 'false', 'half-true'])] # There are some few outliers.
    
    # Prepare Round Df
    round_1_df = df[df['round'] == 0]
    round_2_df = df[df['round'] == 1]
    round_3_df = df[df['round'] == 2]
    
    # Generate balanced test dataset
    minimum_count = min(len(round_1_df), len(round_2_df), len(round_3_df))
    
    # Sample with same amount
    round_1_df = round_1_df.sample(n=minimum_count, random_state=RANDOM_SEED).get(['generated_claims', 'fact_checking_evidences', 'generated_labels'])
    round_2_df = round_2_df.sample(n=minimum_count, random_state=RANDOM_SEED).get(['generated_claims', 'fact_checking_evidences', 'generated_labels'])
    round_3_df = round_3_df.sample(n=minimum_count, random_state=RANDOM_SEED).get(['generated_claims', 'fact_checking_evidences', 'generated_labels'])
    mixed_df = df.sample(n=minimum_count, random_state=RANDOM_SEED).get(['generated_claims', 'fact_checking_evidences', 'generated_labels'])
    
    minimum_count = min(
        len(round_1_df[round_1_df['generated_labels'] == 'true']),
        len(round_1_df[round_1_df['generated_labels'] == 'half-true']),
        len(round_1_df[round_1_df['generated_labels'] == 'false']),
        len(round_2_df[round_2_df['generated_labels'] == 'true']),
        len(round_2_df[round_2_df['generated_labels'] == 'half-true']),
        len(round_2_df[round_2_df['generated_labels'] == 'false']),
        len(round_3_df[round_3_df['generated_labels'] == 'true']),
        len(round_3_df[round_3_df['generated_labels'] == 'half-true']),
        len(round_3_df[round_3_df['generated_labels'] == 'false']),
        len(mixed_df[mixed_df['generated_labels'] == 'true']),
        len(mixed_df[mixed_df['generated_labels'] == 'half-true']),
        len(mixed_df[mixed_df['generated_labels'] == 'false']),
        len(original_df[original_df['label'] == 'true']),
        len(original_df[original_df['label'] == 'half-true']),
        len(original_df[original_df['label'] == 'false']),
    )
    
    round_1_df = format_generated_dataset(round_1_df, minimum_count, RANDOM_SEED)
    round_2_df = format_generated_dataset(round_2_df, minimum_count, RANDOM_SEED)
    round_3_df = format_generated_dataset(round_3_df, minimum_count, RANDOM_SEED)
    mixed_df = format_generated_dataset(mixed_df, minimum_count, RANDOM_SEED)
    original_df = format_original_dataset(original_df, minimum_count, RANDOM_SEED)
    
    round_1_df.to_json(os.path.join(SAVE_PATH, 'round_1.json'))
    round_2_df.to_json(os.path.join(SAVE_PATH, 'round_2.json'))
    round_3_df.to_json(os.path.join(SAVE_PATH, 'round_3.json'))
    mixed_df.to_json(os.path.join(SAVE_PATH, 'mixed.json'))
    original_df.to_json(os.path.join(SAVE_PATH, 'original.json'))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Politifact Link Extractor. Extract Personalities and Article Links from targeted URLs')
    parser.add_argument('--save_path', type=str, default=os.path.join('script_outputs', 'role_playing_misinformation_generated_datasets'), help='Script output location')
    parser.add_argument('--generated_data_path', type=str, default=os.path.join('script_outputs', 'role_playing_misinformation_labelling_output', 'role_playing_label_outputs.json'), help='politifact_articles_links.csv location')
    parser.add_argument('--original_data_path', type=str, default=os.path.join('script_outputs', 'train_dev_test_split'), help='politifact_articles_links.csv location')
    parser.add_argument('--random_seed', type=int, default=42)
    args = parser.parse_args()
    generate_test_dataset(args)