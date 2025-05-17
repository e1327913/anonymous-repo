import re
import os
import argparse
import json
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from bs4 import BeautifulSoup
from openai import OpenAI
import json_repair
# importing necessary functions from dotenv library
from dotenv import load_dotenv, dotenv_values 
# loading variables from .env file
load_dotenv()

def check_current_batch_status(args):
    SAVE_PATH = args.save_path
    OPENAI_BATCH_REQUEST = args.openai_batch_request
    request_file_path = os.path.join(OPENAI_BATCH_REQUEST, 'output.json')
    df = pd.read_json(path_or_buf=request_file_path, lines=True)

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    # Save current args to output
    with open(f"{SAVE_PATH}/argsparse_config.json", 'w') as file:
        json.dump(vars(args), file)
        file.close()     
        
    client = OpenAI(api_key=os.getenv("OPENAI_KEY"))

    details = []
    with open(os.path.join(SAVE_PATH, 'openai_response.jsonl'), 'w') as outfile:
        for row in tqdm(df.iloc()):
            batch_details = client.batches.retrieve(row['batch_id'])
            if batch_details.status == 'completed':
                file_response = client.files.content(batch_details.output_file_id)
                json_output = json_repair.loads(file_response.text)
                print(f"JSON Output: {json_output == None}")
                details.append(json_output)
                df = pd.DataFrame(details)
                df.to_json(os.path.join(SAVE_PATH, 'dataframe.json'))
                json.dump(file_response.text, outfile)
                outfile.write('\n')
            else:
                print(f"Status: {batch_details.status}")
                print(f"Reason: {batch_details.errors}")
        outfile.close()

if __name__ == '__main__':
    timestamp = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    parser = argparse.ArgumentParser(description='Politifact Link Extractor. Extract Personalities and Article Links from targeted URLs')
    parser.add_argument('--save_path', type=str, default=os.path.join('script_outputs', 'politifact-gpt-4o-mini-extractor-results'), help='Script output location')
    parser.add_argument('--openai_batch_request', type=str, default=os.path.join('script_outputs', 'politifact-gpt-4o-mini-extractor-requests'), help='politifact_articles_links.csv location')

    args = parser.parse_args()
    check_current_batch_status(args)