import os
import argparse
import logging
import datetime
import json
import pandas as pd
import tiktoken
import json_repair
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
from roles import DEMOCRAT_ROLE, REPUBLICAN_ROLE, MODERATE_ROLE

# Load variables from .env file
load_dotenv()
# Prepare Logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def consistency_prompt(claim, claim_owner, role_description, intent):
    return f"""
    You will be given a "Claim", "Role", "Role Description", and "Intent".
    Your task is to assess how well does the Claim align with the Role's beliefs and intention.
    
    Provide a rating based on the following Likert scale:
    - 1: Completely Inconsistent: The claim opposes or has no connection to the role.
    - 2: Mostly Inconsistent: The claim contradicts some role beliefs but has a weak connection.
    - 3: Somewhat Consistent: The claim partly aligns but lacks key points or misrepresents intent.
    - 4: Mostly Consistent: The claim follows the role and intent but may miss small details.
    - 5: Perfectly Consistent: The claim fully aligns with the role's beliefs, tone, and intent.

    Definitions:
    - Claim: An unverified statement based on the claim owner's beliefs.
    - Role: A persona used to generate the claim.
    - Role Description: A description of the persona’s personality, beliefs, and worldview.
    - Intent: The claim owner's purpose or motivation behind the statement.

    Response Format:
    {{
        "Question": "Q1",
        "Answer": X,  # (1 to 5)
        "Explanation": "Y"
    }}

    Details:
    - Claim: {claim}
    - Role: {claim_owner}
    - Intent: {intent}
    - Role Description:
        {role_description}
    """

def relevancy_prompt(claim, sources, previous_claims):
    return f"""
    You will be given a "Claim", "Sources", and "Previous Claims".
    Your task is to assess how relevant is the Claim compared to the provided sources and previous claims.
    
    Provide a rating based on the following Likert scale:
    - 1: Completely Irrelevant: The claim does not relate to any sources or previous claims.
    - 2: Weakly Relevant: The claim has a weak or indirect connection to the sources or previous claims.
    - 3: Somewhat Relevant: Somewhat Relevant: The claim mentions sources or previous claims but misinterprets or lacks connections.
    - 4: Mostly Relevant: The claim follows sources or previous claims but misses small details.
    - 5: Perfectly Relevant: The claim fully integrates key facts from sources and previous claims.

    Definitions:
    - Claim: An unverified statement based on the claim owner's beliefs.
    - Sources: Information used to generate the claim.
    - Previous Claims: Earlier claims that contributed to this claim.

    Response Format:
    {{
        "Question": "Q2",
        "Answer": X,  # (1 to 5)
        "Explanation": "Y"
    }}

    Details:
    - Claim: {claim}
    - Sources: 
        {sources}
    - Previous Claims: {previous_claims}
    """

def fluency_prompt(claim):
    return f"""
    You will be given a "Claim".
    Your task is to evaluate how fluent the Claim is in terms of grammar, clarity, and readability.
    
    Provide a rating based on the following Likert scale:
    - 1: Very Poor: Frequent errors make the claim difficult to comprehend.
    - 2: Poor: Contains multiple errors that make it harder to understand.
    - 3: Adequate: Readable but has noticeable errors or awkward phrasing.
    - 4: Good: Mostly correct, with minor errors that do not affect readability.
    - 5: Excellent: Clear, well-written, and grammatically perfect.

    Definitions:
    - Claim: An unverified statement based on the claim owner's beliefs.

    Response Format:
    {{
        "Question": "Q3",
        "Answer": X,  # (1 to 5)
        "Explanation": "Y"
    }}

    Details:
    - Claim: {claim}
    """

def factuality_prompt(claim, sources, previous_claims, evidences):
    return f"""
    You will be given a "Claim", "Sources", "Previous Claims", and "Evidence".
    Your task is to determine how factually correct is the Claim.
    
    Provide a rating based on the following Likert scale:
    - 1: Completely Inaccurate: Completely false or highly misleading.
    - 2: Mostly Inaccurate: Many errors or missing key details, making it misleading.
    - 3: Partially Accurate: Some parts are true, but others are misleading or missing key facts.
    - 4: Mostly Accurate: Mostly factual, with small mistakes or missing details that don’t change the meaning.
    - 5: Completely Accurate: Fully factual, with no misleading parts or missing context.

    Definitions:
    - Claim: An unverified statement based on the claim owner's beliefs.
    - Sources: Information used to generate the claim.
    - Previous Claims: Earlier claims that contributed to this claim.
    - Evidence: Verified information from reliable sources used to assess claim accuracy.

    Response Format:
    {{
        "Question": "Q4",
        "Answer": X,  # (1 to 5, based on the Likert scale)
        "Explanation": "Y"
    }}

    Details:
    - Claim: {claim}
    - Sources: 
        {sources}
    - Previous Claims: 
        {previous_claims}
    - Evidence: 
        {evidences}
    """


def label_assignment_prompt(claim, sources, previous_claims, evidences):
    return f"""
    You will be given a "Claim", "Sources", "Previous Claims", and "Evidence".
    Your task is to determine which label is suitable for the Claim based on the evidence.

    Labels and Definitions:
    - true: Accurate.
    - half-true: Partially accurate, lacks important details or is misleading.
    - false: Inaccurate.

    How to evaluate the label:
    - Carefully analyze whether the claim is fully, partially or not supported by the evidence.
    - Identify specific factual elements in the claims that are supported or contradicted.
    - Note any missing context, exaggerations, or misleading aspects of the claim.
    - Do not make assumptions beyond what the evidence explicitly states.
    - Assign confidence scores to all three labels (true, half-true, false).
    - Select the label with the highest confidence score.

    Response Format:
    {{
        "Question": "Q5",
        "New Label": "true" | "half-true" | "false"
        "Confidence Scores": {{
            "true": X,  # Confidence score (0-1)
            "half-true": Y,  # Confidence score (0-1)
            "false": Z  # Confidence score (0-1)
        }},
        "Explanation": "Detailed reasoning for your decision."
    }}

    Details:
    - Claim: {claim}
    - Sources: {sources}
    - Previous Claims: {previous_claims}
    - Evidence: {evidences}
    """

def perform_gpt_metrics(args):
    # Argparse
    SAVE_PATH = args.save_path
    DATASET_PATH = args.dataset_path
    
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    # Save current args to output
    with open(f"{SAVE_PATH}/argsparse_run_config.json", 'w') as file:
        json.dump(vars(args), file)
        file.close()
        
    rpg_df = pd.read_csv(DATASET_PATH)
    client = OpenAI(api_key=os.getenv("OPENAI_KEY"))
    
    prompts = {
        'Q1': [],
        'Q2': [],
        'Q3': [],
        'Q4': [],
        'Q5': [],
    }
    
    for row in rpg_df.iloc():
        evidences = row['summarized_evidence']
        claim = row['generated_claim']
        intent = row['generated_intent']
        claim_owner = row['role']
        round = row['round']
        role_description = row['role_description']
        sources = row['summarized_sources']
        previous_claims = None if pd.isna(row['previous_claims']) else row['previous_claims']
        if round == 1:
            previous_claims = f"- {previous_claims}\n"
        elif round == 2:
            previous_claims = '\n'.join([f"- {previous_claim}" for previous_claim in previous_claims.split('@')])
        
        prompts['Q1'].append({'url': f"Q1_{claim_owner}_{round}_{row['url']}_{row['role_sequence']}", "prompt": consistency_prompt(claim, claim_owner, role_description, intent)})
        prompts['Q2'].append({'url': f"Q2_{claim_owner}_{round}_{row['url']}_{row['role_sequence']}", "prompt": relevancy_prompt(claim, sources, previous_claims)})
        prompts['Q3'].append({'url': f"Q3_{claim_owner}_{round}_{row['url']}_{row['role_sequence']}", "prompt": fluency_prompt(claim)})
        prompts['Q4'].append({'url': f"Q4_{claim_owner}_{round}_{row['url']}_{row['role_sequence']}", "prompt": factuality_prompt(claim, sources, previous_claims, evidences)})
        prompts['Q5'].append({'url': f"Q5_{claim_owner}_{round}_{row['url']}_{row['role_sequence']}", "prompt": label_assignment_prompt(claim, sources, previous_claims, evidences)})

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
                "description": f"Politifact Misinformation Sources & Extraction with file {file_path}"
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
    parser.add_argument('--save_path', type=str, default=os.path.join('script_outputs', 'gpt_4o_mini_questionnaire_metric-full'), help='Script output location')
    parser.add_argument('--dataset_path', type=str, default=os.path.join('script_outputs', 'questionnaire', 'summarized_content', 'summarized_questionnaire_data.csv'))
    args = parser.parse_args()
    perform_gpt_metrics(args)