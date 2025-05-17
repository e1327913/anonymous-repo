import os
import argparse
import logging
import datetime
import json
import pandas as pd
import json_repair
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Load variables from .env file
load_dotenv()
# Prepare Logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

maps = {
    'Q1': 'Role Playing Consistency: How consistent the "Claim" is with the the "Role", "Role Description", and "Intent"?',
    'Q2': 'Relevancy: How relevant the "Claim" is to the provided "Sources" and "Previous Claims"',
    'Q3': 'Fluency: How fluent the "Claim" is in terms of grammar, clarity, and readability',
    'Q4': 'Factuality: Does the "Claim" contains inaccuracies or missing information',
    'Q5': 'Label Assignment: Is the assigned "Label" correct for the Claim?'
}


def retrieve_gpt_metrics(args):
    # Argparse
    SAVE_PATH = args.save_path
    
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    # Save current args to output
    with open(f"{SAVE_PATH}/argsparse_config.json", 'w') as file:
        json.dump(vars(args), file)
        file.close()
    
    df = pd.read_json(os.path.join(SAVE_PATH, 'dataframe.json'))
    df = df.groupby('question_key')
    for (question_key, group) in df:
        answer_map = None
        if question_key == 'Q1':
            answer_map = {
                1: '1 - Completely Inconsistent',
                2: '2 - Mostly Inconsistent', 
                3: '3 - Moderately Consistent',
                4: '4 - Mostly Consistent',
                5: '5 - Perfectly Consistent',
            }
            group['answer'] = group['answer'].map(lambda x: answer_map[x])

        if question_key == 'Q2':
            answer_map = {
                1: '1 - Completely Irrelevant',
                2: '2 - Weakly Relevant', 
                3: '3 - Moderately Relevant',
                4: '4 - Mostly Relevant',
                5: '5 - Perfectly Relevant',
            }
            group['answer'] = group['answer'].map(lambda x: answer_map[x])

        if question_key == 'Q3':
            answer_map = {
                1: '1 - Very Poor',
                2: '2 - Poor', 
                3: '3 - Moderate',
                4: '4 - Good',
                5: '5 - Excellent',
            }
            group['answer'] = group['answer'].map(lambda x: answer_map[x])

        if question_key == 'Q4':
            answer_map = {
                1: '1 - Completely Inaccurate',
                2: '2 - Mostly Inaccurate', 
                3: '3 - Partially Accurate',
                4: '4 - Mostly Accurate',
                5: '5 - Completely Accurate',
            }
            group['answer'] = group['answer'].map(lambda x: answer_map[x])

        data = group['answer'].value_counts(sort=True)
        data = data.sort_index()

        _, ax = plt.subplots(figsize=(13, 9))
        data.plot.barh(ax=ax)
        ticks = list(ax.get_xticks())
        ticks = ticks + [ticks[-1] + 1000]
        ax.set_xticks(ticks)
        plt.xticks(rotation=35)
        plt.yticks(rotation=35)
        plt.title(f"{maps[question_key]}. Total: ({len(group)})")
        for index, value in enumerate(data):
            ax.text(value + (max(data) * 0.02), index, f"{value} ({(value / len(group)) * 100:.2f}%)", va='center')
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_PATH, f'{question_key}.png'))
        plt.close()

        if question_key == 'Q4':
            _, ax = plt.subplots(figsize=(14, 9))
            filtered_group = group[['role', 'answer']].groupby('role').value_counts(sort=False)
            filtered_group.plot.barh(ax=ax)
            plt.title(f'Q4 Validity  Total: ({len(group)})')
            for index, value in enumerate(filtered_group):
                ax.text(value + (max(data) * 0.002), index, f"{value} ({(value / (len(group))) * 100:.2f}%)", va='center')
            ticks = list(ax.get_xticks())
            ticks = ticks + [ticks[-1] + 200]
            ax.set_xticks(ticks)
            plt.tight_layout()
            plt.savefig(os.path.join(SAVE_PATH, f'q4_validity.png'))
            plt.close()
    
if __name__ == '__main__':
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    parser = argparse.ArgumentParser(description='Politifact Link Extractor. Extract Personalities and Article Links from targeted URLs')
    parser.add_argument('--save_path', type=str, default=os.path.join('script_outputs', 'gpt_4o_mini_metric_request'), help='Script output location')
    args = parser.parse_args()
    retrieve_gpt_metrics(args)