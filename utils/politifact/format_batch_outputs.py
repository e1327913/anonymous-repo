import os
import argparse
import json
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import json_repair

def remove_blacklisted_sentences(data):
    blacklisted_phrases = [
        'Read more',
        'flagged as part of',
        'fact-checking process and rating system.',
        'Screengrab from',
        'to combat false news and misinformation on its News Feed',
    ]
    cleaned_data = []
    for sentence in data:
        assert isinstance(sentence, str) == True
        is_clean = True
        for phrase in blacklisted_phrases:
            if phrase in sentence:
                is_clean = False
        
        if is_clean:
            cleaned_data.append(sentence)

    return cleaned_data

def format_openai_output(args):
    SAVE_PATH = args.save_path
    OPENAI_BATCH_RESULTS = args.openai_batch_results
    POLITIFACT_EXTRACTED_ARTICLES_FOLDER = args.politifact_extracted_articles
    request_file_path = os.path.join(OPENAI_BATCH_RESULTS, 'dataframe.json')
    openai_request_file_path = os.path.join(OPENAI_BATCH_RESULTS, 'openai_response.jsonl')
    df = pd.read_json(request_file_path)
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    # Save current args to output
    with open(f"{SAVE_PATH}/argsparse_config.json", 'w') as file:
        json.dump(vars(args), file)
        file.close() 

    openai_raw_response = pd.read_json(openai_request_file_path, lines=True)

    dataframe_inputs = []
    for rows in tqdm(openai_raw_response.iloc()):
        items = [json_repair.loads(rows[0])]
        for data in items:
            status_code = data['response']['status_code']
            if status_code != 200:
                raise ValueError(data)
            responses = data['response']['body']['choices']
            for response in responses:
                details = json_repair.loads(response['message']['content'])
                is_valid = True
                required_keys = ['Misinformation Sources', 'Fact-Checking Evidence', 'Transition Sentence']
                for key in required_keys:
                    if isinstance(details, str) or key not in details.keys():
                        is_valid = False
                if response['message']['refusal'] is None and is_valid:
                    misinformation_sources = []
                    for item in details['Misinformation Sources']:
                        if isinstance(item, dict) and len(item.keys()) > 1:
                            current_keys = list(item.keys())
                            current_details = []
                            for key in current_keys[1:]:
                                if isinstance(item[key], str):
                                    current_details.append(item[key])
                                elif isinstance(item[key], list):
                                    current_details.append('.'.join(item[key]))
                            current_string = '.'.join(current_details)
                            source = item[current_keys[0]]
                            if source not in current_string:
                                current_string = f"{source}. {current_string}"
                            misinformation_sources.append(current_string)
                        elif isinstance(item, dict) and len(item.keys()) == 1:
                            current_keys = list(item.keys())
                            misinformation_sources.append(item[current_keys[0]])
                        elif isinstance(item, str):
                            misinformation_sources.append(item)
                        else:
                            print(details['Misinformation Sources'])
                    
                    fact_checking_evidences = []
                    for item in details['Fact-Checking Evidence']:
                        if isinstance(item, dict) and len(item.keys()) > 1:
                            current_keys = list(item.keys())
                            current_details = []
                            for key in current_keys[1:]:
                                if isinstance(item[key], str):
                                    current_details.append(item[key])
                                elif isinstance(item[key], list):
                                    cleaned = []
                                    for item_string in item[key]:
                                        if isinstance(item_string, str):
                                            cleaned.append(item_string)
                                    current_details.append('.'.join(item_string))
                            current_string = '.'.join(current_details)
                            source = item[current_keys[0]]
                            if source not in current_string:
                                current_string = f"{source}. {current_string}"
                            fact_checking_evidences.append(current_string)
                        elif isinstance(item, dict) and len(item.keys()) == 1:
                            current_keys = list(item.keys())
                            fact_checking_evidences.append(item[current_keys[0]])
                        elif isinstance(item, str):
                            fact_checking_evidences.append(item)
                        else:
                            raise ValueError(item)

                    method_key = 'Method'
                    if 'Selected Method' in details.keys():
                        method_key = 'Selected Method'
                    elif 'selected_method' in details.keys():
                        method_key = 'selected_method'
                    elif 'method' in details.keys():
                        method_key = 'method'
                        
                    # Remove the Flagged sentence in misinformation sources and fact-checking evidences
                    cleaned_misinformation_sources = remove_blacklisted_sentences(misinformation_sources)
                    cleaned_fact_checking_evidences = remove_blacklisted_sentences(fact_checking_evidences)

                    dataframe_inputs.append({
                        'url': data['custom_id'],
                        'misinformation_sources': cleaned_misinformation_sources,
                        'fact_checking_evidences': cleaned_fact_checking_evidences,
                        'method': details[method_key] if method_key in details.keys() else None,
                        'transition_sentence': details['Transition Sentence'] if len(details.keys()) == 4 and 'Transition Sentence' in details.keys() else None,
                        'is_usable': len(cleaned_misinformation_sources) > 0 and len(cleaned_fact_checking_evidences) > 0
                    })

    metadata = os.path.join(POLITIFACT_EXTRACTED_ARTICLES_FOLDER, 'politifact_extracted_articles.json')
    df = pd.read_json(metadata)
    df['datetime'] = pd.to_datetime(df['claim_date'])
    new_df = pd.DataFrame(dataframe_inputs)
    df = df.merge(new_df,on='url')
    df.to_json(os.path.join(SAVE_PATH, 'formatted_openai_output.json'))

if __name__ == '__main__':
    timestamp = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    parser = argparse.ArgumentParser(description='Politifact Link Extractor. Extract Personalities and Article Links from targeted URLs')
    parser.add_argument('--save_path', type=str, default=os.path.join('script_outputs', 'politifact-gpt-4o-mini-formatted-results'), help='Script output location')
    parser.add_argument('--openai_batch_results', type=str, default=os.path.join('script_outputs', 'politifact-gpt-4o-mini-extractor-results'), help='politifact_articles_links.csv location')
    parser.add_argument('--politifact_extracted_articles', type=str, default=os.path.join('script_outputs', 'politifact-extracted-articles'), help='politifact_articles_links.csv location')

    args = parser.parse_args()
    format_openai_output(args)