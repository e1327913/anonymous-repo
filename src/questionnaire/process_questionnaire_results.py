import pandas as pd 
import os 
import datetime
import argparse
import json

def extract_xlsx_results(args):
    # Argparse
    SAVE_PATH = args.save_path
    DATASET_PATH = args.dataset_path
    EXCEL_PATH = args.excel_file
    
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    # Save current args to output
    with open(f"{SAVE_PATH}/argsparse_run_config.json", 'w') as file:
        json.dump(vars(args), file)
        file.close()

    df = pd.read_csv(DATASET_PATH)[:15]

    q1_results = []
    q2_results = []
    q3_results = []
    q4_results = []
    q5_results = []

    excel_df = pd.read_excel(EXCEL_PATH, sheet_name=None)
    for sheet_name in excel_df.keys():
        if sheet_name != 'questionnaire_data':
            current_df = excel_df[sheet_name]
            for key in current_df:
                if 'Question 1' in key:
                    q1_results.append(current_df[key].iloc[0].split('-')[0].strip())
                elif 'Question 2' in key:
                    q2_results.append(current_df[key].iloc[0].split('-')[0].strip())
                elif 'Question 3' in key:
                    q3_results.append(current_df[key].iloc[0].split('-')[0].strip())
                elif 'Question 4' in key:
                    q4_results.append(current_df[key].iloc[0].split('-')[0].strip())
                elif 'Question 5' in key:
                    q5_results.append(current_df[key].iloc[0].split(':')[0].strip())
    
    df['q1'] = q1_results
    df['q2'] = q2_results
    df['q3'] = q3_results
    df['q4'] = q4_results
    df['q5'] = q5_results

    df.to_json(os.path.join(SAVE_PATH, 'questionnaire_parsed_results.json'))

if __name__ == '__main__':
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    parser = argparse.ArgumentParser(description='Politifact Link Extractor. Extract Personalities and Article Links from targeted URLs')
    parser.add_argument('--save_path', type=str, default=os.path.join('script_outputs', 'questionnaire_results'), help='Script output location')
    parser.add_argument('--dataset_path', type=str, default=os.path.join('script_outputs', 'questionnaire', 'questionnaire_data.csv'))
    parser.add_argument('--excel_file', type=str, default=os.path.join('src', 'questionnaire', 'questionnaire_data.xlsx'))
    args = parser.parse_args()
    extract_xlsx_results(args)