import os
import argparse
import json
import pandas as pd
import tiktoken
import uuid
from prompts.evaluation_prompts import zero_shot_evaluation_prompt, zero_shot_evaluation_prompt_cot, few_shot_evaluation_prompt, few_shot_evaluation_prompt_cot
from openai import OpenAI
from tqdm.auto import tqdm
# importing necessary functions from dotenv library
from dotenv import load_dotenv
# loading variables from .env file
load_dotenv()

def prepare_evaluation_prompts(dataframe, mode='zero_shot'):
    prompts = []
    for row in dataframe.iloc():
        claim = row['claim']
        fact_checking_evidences = row['fact_checking_evidence']
        fce_dict = {}
        for index, text in enumerate(fact_checking_evidences):
            fce_dict[f"fce_source_{index}"] = text
        prompt = None
        if mode == 'zero_shot':
            prompt = zero_shot_evaluation_prompt(claim, fce_dict)
        elif mode == 'few_shot':
            prompt = few_shot_evaluation_prompt(claim, fce_dict)
        elif mode == 'zero_shot_cot':
            prompt = zero_shot_evaluation_prompt_cot(claim, fce_dict)
        elif mode == 'few_shot_cot':
            prompt = few_shot_evaluation_prompt_cot(claim, fce_dict)
        else:
            raise ValueError(mode)
        prompts.append({'id': row['id'], 'prompt': prompt})
    return prompts

def start_evaluation(args):
    # Argparse
    MODE = args.mode
    SAVE_PATH = os.path.join(args.save_path)
    EVAL_DATA_PATH = args.eval_data_path
    BATCH_SIZE = args.batch_size
    
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    # Save current args to output
    with open(f"{SAVE_PATH}/argsparse_config.json", 'w') as file:
        json.dump(vars(args), file)
        file.close()
        
    output_file_names = []
    client = OpenAI(api_key=os.getenv("OPENAI_KEY"))
    encoder = tiktoken.encoding_for_model('gpt-4o-mini')
    dataframe = pd.read_json(EVAL_DATA_PATH)
    prompts = prepare_evaluation_prompts(dataframe, MODE)
    assert len(prompts) == len(dataframe)
    gap=BATCH_SIZE
    
    for index in range(0, len(prompts), gap):
        texts = 0
        current_prompts = prompts[index : index + gap]
        request_file_path = os.path.join(SAVE_PATH, f'eval_requests_{index}.jsonl')
        with open(request_file_path, 'w') as outfile:
            for index, prompt_data in enumerate(current_prompts):
                df_id, prompt = prompt_data['id'], prompt_data['prompt']
                data = {
                    "custom_id": f"{str(uuid.uuid4())}@{df_id}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-4o-mini",
                        "messages": [
                            {"role": "user", "content": prompt},
                        ],
                    }
                }
                tokens = encoder.encode(prompt)
                texts += len(tokens)
                json.dump(data, outfile)
                outfile.write('\n')
            outfile.close()
        output_file_names.append(request_file_path)
    
    final_details = []
    for index, file_path in tqdm(enumerate(output_file_names)):
        batch_input_file = client.files.create(
            file=open(file_path, "rb"),
            purpose="batch"
        )
        batch_input_file_id = batch_input_file.id
        output = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": f"Politifact Misinformation Sources & Fact-Checking Extraction with file {file_path}"
            }
        )
        print(output)
        final_details.append({
            "batch_id": output.id,
            "original_file": file_path
        })

    with open(os.path.join(SAVE_PATH, 'output.json'), 'w') as outfile:
        for detail in tqdm(final_details):
            json.dump(detail, outfile)
            outfile.write('\n')
        outfile.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Decoder Evaluation')
    
    # Data Arguments
    parser.add_argument("--save_path", type=str, default=os.path.join('script_outputs', 'decoder_models_results', 'gpt-4o-mini'))
    parser.add_argument("--eval_data_path", type=str, default=os.path.join('script_outputs', 'role_playing_misinformation_generated_datasets', '2025-03-21-00-48-13', 'mixed.json'))
    parser.add_argument('--mode', type=str, default='few_shot', help='Script output location')
    parser.add_argument('--batch_size', type=int, default=2000, help='Script output location')
    
    args = parser.parse_args()
    start_evaluation(args)