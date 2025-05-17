import os
import argparse
import pandas as pd
import json
import logging
import math
from datetime import datetime

# Prepare Logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def train_dev_test(args):
    SAVE_PATH = args.save_path
    DATA_PATH = args.data_path
    RANDOM_SEED = args.random_seed
    data_path = os.path.join(DATA_PATH, 'formatted_openai_output.json')

    df = pd.read_json(data_path)

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    # Save current args to output
    with open(f"{SAVE_PATH}/argsparse_config.json", 'w') as file:
        json.dump(vars(args), file)
        file.close() 

    # Beautify the dataset
    df['claim'] = df['claim'].apply(lambda x: x.strip())
    df['claim_owner'] = df['claim_owner'].apply(lambda x: x.strip())
    print(df['datetime'].max())

    logging.info(f"Data Distribution Count: \n {df['label'].value_counts()}")

    # Replace because PolitiFact has updated it
    df['label'] = df['label'].replace(to_replace='barely-true', value='mostly-false')
    df['label'] = df['label'].replace(to_replace='pants-fire', value='pants-on-fire')

    # Shorten Dataset Labels to True, False, and Half-True
    df['label'] = df['label'].replace(to_replace='pants-on-fire', value='false')
    df['label'] = df['label'].replace(to_replace='mostly-false', value='false')
    df['label'] = df['label'].replace(to_replace='mostly-true', value='half-true')

    assert len(set(df['label'])) == 3
    logging.info(f"Data Distribution Count: \n {df['label'].value_counts()}")

    # Data Distribution Counts
    # false        13527
    # half-true     6618
    # true          2263

    minimum_count = len(df[df['label'] == 'true'])

    sampled_df = df.groupby('label').sample(n=minimum_count, random_state=RANDOM_SEED)
    test = sampled_df.groupby('label').sample(n=math.floor(0.1 * minimum_count), random_state=RANDOM_SEED)
    train_dev_df = sampled_df.drop(test.index)
    dev = train_dev_df.groupby('label').sample(n=math.floor(0.1 * minimum_count), random_state=RANDOM_SEED)
    train = train_dev_df.drop(dev.index)

    logging.info(f"Train Length: {len(train)} Train Labels: {set(train.label)} Label Counts: {train['label'].value_counts()}")
    logging.info(f"Dev Length: {len(dev)} Dev Labels: {set(dev.label)} Label Counts: {dev['label'].value_counts()}")
    logging.info(f"Test Length: {len(test)} Test Labels: {set(test.label)} Label Counts: {test['label'].value_counts()}")

    logging.info(f"Save train dev test split")

    train.to_json(os.path.join(SAVE_PATH, "train.json"))
    dev.to_json(os.path.join(SAVE_PATH, "dev.json"))
    test.to_json(os.path.join(SAVE_PATH, "test.json"))

if __name__ == '__main__':
    timestamp = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    parser = argparse.ArgumentParser(description='Train Dev Test Dataset Split')
    parser.add_argument('--save_path', type=str, default=os.path.join('script_outputs', 'train_dev_test_split'), help='Script output location')
    parser.add_argument('--data_path', type=str, default=os.path.join('script_outputs', 'politifact-gpt-4o-mini-formatted-results'), help='Data Location')
    parser.add_argument('--random_seed', type=int, default=42)
    args = parser.parse_args()
    train_dev_test(args)