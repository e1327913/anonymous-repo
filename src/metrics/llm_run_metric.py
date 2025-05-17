import os
import re
import argparse
import logging
import datetime
import json
import pandas as pd
import tiktoken
import json_repair
from dotenv import load_dotenv
from openai import OpenAI
from collections import Counter
from tqdm import tqdm
from roles import DEMOCRAT_ROLE, REPUBLICAN_ROLE, MODERATE_ROLE

# Load variables from .env file
load_dotenv()
# Prepare Logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Align with Questionnaire
def consistency_prompt(claim, claim_owner, role_description, intent):
    return f"""How well does the Claim align with the Role's beliefs and intention?

    Details:
    - Role: {claim_owner}
    - Claim: {claim}
    - Role Description: 
        {role_description}
    - Intent: {intent}
    
    Provide a rating based on the following Likert scale:
    - 5: Perfectly Consistent: The claim fully aligns with the role's beliefs, tone, and intent.
    - 4: Mostly Consistent: The claim follows the role and intent but may miss small details.
    - 3: Somewhat Consistent: The claim partly aligns but lacks key points or misrepresents intent.
    - 2: Mostly Inconsistent: The claim contradicts some role beliefs but has a weak connection.
    - 1: Completely Inconsistent: The claim opposes or has no connection to the role.

    Response Format:
    {{
        "Question": "Q1",
        "Answer": X,  # (1 to 5)
        "Explanation": "Y"
    }}
    """

def relevancy_prompt(claim, sources, previous_claims):
    return f"""How relevant is the Claim compared to the provided sources and previous claims?
    
    Details:
    - Claim: {claim}
    - Sources: 
        {sources}
    - Previous Claims: 
        {previous_claims}

    Provide a rating based on the following Likert scale:
    - 5: Perfectly Relevant: The claim fully integrates key facts from sources and previous claims.
    - 4: Mostly Relevant: The claim follows sources or previous claims but misses small details.
    - 3: Somewhat Relevant: The claim mentions sources or previous claims but misinterprets or lacks connections.
    - 2: Weakly Relevant: The claim has a weak or indirect connection to the sources or previous claims.
    - 1: Completely Irrelevant: The claim does not relate to any sources or previous claims.

    Response Format:
    {{
        "Question": "Q2",
        "Answer": X,  # (1 to 5)
        "Explanation": "Y"
    }}

    """

def fluency_prompt(claim):
    return f"""How fluent the Claim is in terms of grammar, clarity, and readability?
    
    Details:
    - Claim: {claim}

    Provide a rating based on the following Likert scale:
    - 5: Excellent: Clear, well-written, and grammatically perfect.
    - 4: Good: Mostly correct, with minor errors that do not affect readability.
    - 3: Adequate: Readable but has noticeable errors or awkward phrasing.
    - 2: Poor: Contains multiple errors that make it harder to understand.
    - 1: Very Poor: Frequent errors make the claim difficult to comprehend.
      
    Response Format:
    {{
        "Question": "Q3",
        "Answer": X,  # (1 to 5)
        "Explanation": "Y"
    }}
    """

def factuality_prompt(claim, sources, previous_claims, evidences):
    return f"""How factually correct is the Claim?

    Details:
    - Claim: {claim}
    - Sources: 
        {sources}
    - Previous Claims: 
        {previous_claims}
    - Evidence: 
        {evidences}
    
    Provide a rating based on the following Likert scale:
    - 5: Completely Accurate: Fully factual, with no misleading parts or missing context.
    - 4: Mostly Accurate: Mostly factual, with small mistakes or missing details that don’t change the meaning.
    - 3: Partially Accurate: Some parts are true, but others are misleading or missing key facts.
    - 2: Mostly Inaccurate: Many errors or missing key details, making it misleading.
    - 1: Completely Inaccurate: Completely false or highly misleading.
          
    Response Format:
    {{
        "Question": "Q4",
        "Answer": X,  # (1 to 5, based on the Likert scale)
        "Explanation": "Y"
    }}
    """


def label_assignment_prompt(claim, evidences):
    return f"""Which label do you think is suitable for the Claim based on the evidence?

    Details:
    - Claim: {claim}
    - Evidence: 
        {evidences}

    Labels and Definitions:
    - true: Fully accurate.
    - half-true: Partially accurate, lack important details or is misleading.
    - false: Inaccurate.

    Response Format:
    {{
        "Question": "Q5",
        "Answer": "true" | "half-true" | "false",  # Select the most appropriate label for this claim.
        "Explanation": "Detailed reasoning for your decision."
    }}
    """

# def consistency_prompt(claim, claim_owner, role_description, intent):
#     return f"""
#     Your task is to answer this question:
#     How well does the Claim align with the Role's beliefs and intention?

#     Details:
#     - Role: {claim_owner}
#     - Claim: {claim}
#     - Role Description: 
#         {role_description}
#     - Intent: {intent}
    
#     Provide a rating based on the following Likert scale:
#     - 5: Perfectly Consistent: The claim fully aligns with the role's beliefs, tone, and intent.
#     - 4: Mostly Consistent: The claim follows the role and intent but may miss small details.
#     - 3: Somewhat Consistent: The claim partly aligns but lacks key points or misrepresents intent.
#     - 2: Mostly Inconsistent: The claim contradicts some role beliefs but has a weak connection.
#     - 1: Completely Inconsistent: The claim opposes or has no connection to the role.

#     Response Format:
#     {{
#         "Question": "Q1",
#         "Answer": X,  # (1 to 5)
#         "Explanation": "Y"
#     }}
#     """

# def relevancy_prompt(claim, sources, previous_claims):
#     return f"""
#     Your task is to answer this question:
#     How relevant is the Claim compared to the provided sources and previous claims?
    
#     Details:
#     - Claim: {claim}
#     - Sources: 
#         {sources}
#     - Previous Claims: 
#         {previous_claims}

#     Provide a rating based on the following Likert scale:
#     - 5: Perfectly Relevant: The claim fully integrates key facts from sources and previous claims.
#     - 4: Mostly Relevant: The claim follows sources or previous claims but misses small details.
#     - 3: Somewhat Relevant: The claim mentions sources or previous claims but misinterprets or lacks connections.
#     - 2: Weakly Relevant: The claim has a weak or indirect connection to the sources or previous claims.
#     - 1: Completely Irrelevant: The claim does not relate to any sources or previous claims.

#     Response Format:
#     {{
#         "Question": "Q2",
#         "Answer": X,  # (1 to 5)
#         "Explanation": "Y"
#     }}

#     """

# def fluency_prompt(claim):
#     return f"""
#     Your task is to answer this question:
#     How fluent the Claim is in terms of grammar, clarity, and readability?
    
#     Details:
#     - Claim: {claim}

#     Provide a rating based on the following Likert scale:
#     - 5: Excellent: Clear, well-written, and grammatically perfect.
#     - 4: Good: Mostly correct, with minor errors that do not affect readability.
#     - 3: Adequate: Readable but has noticeable errors or awkward phrasing.
#     - 2: Poor: Contains multiple errors that make it harder to understand.
#     - 1: Very Poor: Frequent errors make the claim difficult to comprehend.
      
#     Response Format:
#     {{
#         "Question": "Q3",
#         "Answer": X,  # (1 to 5)
#         "Explanation": "Y"
#     }}
#     """

# def factuality_prompt(claim, sources, previous_claims, evidences):
#     return f"""
#     Your task is to answer this question:
#     How factually correct is the Claim?

#     Details:
#     - Claim: {claim}
#     - Sources: 
#         {sources}
#     - Previous Claims: 
#         {previous_claims}
#     - Evidence: 
#         {evidences}
    
#     Provide a rating based on the following Likert scale:
#     - 5: Completely Accurate: Fully factual, with no misleading parts or missing context.
#     - 4: Mostly Accurate: Mostly factual, with small mistakes or missing details that don’t change the meaning.
#     - 3: Partially Accurate: Some parts are true, but others are misleading or missing key facts.
#     - 2: Mostly Inaccurate: Many errors or missing key details, making it misleading.
#     - 1: Completely Inaccurate: Completely false or highly misleading.
          
#     Response Format:
#     {{
#         "Question": "Q4",
#         "Answer": X,  # (1 to 5, based on the Likert scale)
#         "Explanation": "Y"
#     }}
#     """


# def label_assignment_prompt(claim, evidences):
#     return f"""
#     Your task is to answer this question:
#     Which label do you think is suitable for the Claim based on the evidence?

#     Details:
#     - Claim: {claim}
#     - Evidence: 
#         {evidences}

#     Labels and Definitions:
#     - true: Fully accurate.
#     - half-true: Partially accurate, lack important details or is misleading.
#     - false: Inaccurate.

#     Response Format:
#     {{
#         "Question": "Q5",
#         "Answer": "true" | "half-true" | "false",  # Select the most appropriate label for this claim.
#         "Explanation": "Detailed reasoning for your decision."
#     }}
#     """

def perform_gpt_metrics(args):
    # Argparse
    SAVE_PATH = args.save_path
    ORIGINAL_DATA_PATH = args.original_data_path
    DATASET_PATH = args.dataset_path
    
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    # Save current args to output
    with open(f"{SAVE_PATH}/argsparse_run_config.json", 'w') as file:
        json.dump(vars(args), file)
        file.close()
        
    rpg_df = pd.read_csv(DATASET_PATH)
    data_df = pd.read_json(ORIGINAL_DATA_PATH)
    client = OpenAI(api_key=os.getenv("OPENAI_KEY"))
    
    prompts = {
        'Q1': [],
        'Q2': [],
        'Q3': [],
        'Q4': [],
        'Q5': [],
    }
    
    # rpg_df = rpg_df.groupby('round').sample(300)
    
    for idx, row in rpg_df.iterrows():
        evidences = row['summarized_evidence']
        claim = row['generated_claim']
        intent = row['generated_intent']
        claim_owner = row['role']
        round = row['round']
        role_description = row['role_description']
        sources = row['summarized_sources']
        previous_claims = None if isinstance(row['previous_claims'], float) else [f'- {item} \n' for item in row['previous_claims'].split('@')]  
        prompts['Q1'].append({'url': f"Q1_{claim_owner}_{round}_{row['url']}_{row['role_sequence']}", "prompt": consistency_prompt(claim, claim_owner, role_description, intent)})
        prompts['Q2'].append({'url': f"Q2_{claim_owner}_{round}_{row['url']}_{row['role_sequence']}", "prompt": relevancy_prompt(claim, sources, previous_claims)})
        prompts['Q3'].append({'url': f"Q3_{claim_owner}_{round}_{row['url']}_{row['role_sequence']}", "prompt": fluency_prompt(claim)})
        prompts['Q4'].append({'url': f"Q4_{claim_owner}_{round}_{row['url']}_{row['role_sequence']}", "prompt": factuality_prompt(claim, sources, previous_claims, evidences)})
        prompts['Q5'].append({'url': f"Q5_{claim_owner}_{round}_{row['url']}_{row['role_sequence']}", "prompt": label_assignment_prompt(claim, evidences)})

    output_file_names = []
    encoder = tiktoken.encoding_for_model('gpt-4o-mini')
    for key in prompts:
        texts = 0
        current_prompts = prompts[key]
        request_file_path = os.path.join(SAVE_PATH, f'eval_requests_{key}.jsonl')
        with open(request_file_path, 'w') as outfile:
            for prompt_data in current_prompts:
                data = {
                    "custom_id": prompt_data['url'],
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-4o-mini",
                        "messages": [
                            {"role": "user", "content": prompt_data['prompt']},
                        ],
                    }
                }
                tokens = encoder.encode(prompt_data['prompt'])
                texts += len(tokens)
                json.dump(data, outfile)
                outfile.write('\n')
            outfile.close()
        print(texts)
        output_file_names.append(request_file_path)
    
    final_details = []
    for file_path in tqdm(output_file_names):
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
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    parser = argparse.ArgumentParser(description='Politifact Link Extractor. Extract Personalities and Article Links from targeted URLs')
    parser.add_argument('--save_path', type=str, default=os.path.join('script_outputs', 'gpt_4o_mini_metric_request'), help='Script output location')
    parser.add_argument('--original_data_path', type=str, default=os.path.join('script_outputs', 'politifact-gpt-4o-mini-formatted-results', '2025-03-20--09-44-24', 'formatted_openai_output.json'))
    parser.add_argument('--dataset_path', type=str, default=os.path.join('script_outputs', 'questionnaire', 'summarized_content', 'summarized_questionnaire_data.csv'))
    args = parser.parse_args()
    perform_gpt_metrics(args)