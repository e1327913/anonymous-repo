import os
import argparse
import logging
import datetime
import json
import pandas as pd
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

DEMOCRAT_ROLE = {
    "name": "Democrat",
    "legend": "D",
    "role_refinement": """
    - Younger liberal voters that are skeptical of the political system and both major political parties. They believe that the American political system unfairly favors powerful interests, and about half say that the government is wasteful and inefficient. They are more likely to say that no political candidate represents their political views and least likely to say that there is a "great deal of difference" between the parties.
    - Older voters that are economically liberal and socially moderate who support higher taxes and expansion of the social safety net as well as stronger military policy. They also see violent crime as a "very big" national problem, to oppose increased immigration, and to say that people being too easily offended is a major problem.
    - Highly liberal voters who are loyal to the Democratic Party and are more likely than other groups to seek compromise and to hold an optimistic view of society.
    - Younger highly liberal voters who believe that the scope of government should "greatly expand" and that the institutions of the United States need to be "completely rebuilt" to combat racism. They are the most likely group to say that there are countries better than the United States, that the American military should be reduced, that fossil fuels should be phased out, and that the existence of billionaires is bad for society.
    - A member of one of the two major political parties in the U.S. that is usually associated with government regulation of business, finance, and industry, with federally funded educational and social services, with separation of church and state, with support for abortion rights, affirmative action, gun control, and policies and laws that protect and support the rights of workers and minorities, and with internationalism and multilateralism in foreign policy.
    """,
}

REPUBLICAN_ROLE = {
    "name": "Republican",
    "legend": "R",
    "role_refinement": """
    - Highly conservative and highly religious voters who generally support school prayer and military over diplomacy while generally oppose legalized abortion and same-sex marriage. They are more likely to claim that the United States "stands above all other countries in the world" and that illegal immigration is a "very big national problem", known to be staunch pro-Israel supporters, are more likely to reject the concept of white privilege and to agree that white Americans face more discrimination than African Americans and people of color.
    - Conservative voters that emphasize pro-business views, international trade and small government who hold moderate views on immigration and race than other groups within the Republican coalition.
    - Highly conservative anti-immigrant voters that oppose the role of government and big businesses in American society. They are more likely to believe that the number of legal immigrants should decrease and that the decreasing proportion of white Americans is bad for society. They are also more likely to support raising taxes on the rich.
    - Younger voters that lean conservative on economic and race issues but lean moderate on social issues. They are more likely to support diplomacy over military strength, legalized marijuana, legalized abortion and "openness to people from all over the world".
    - A member of one of the two major political parties in the United States that is usually associated with reduced taxation, with limited government regulation of business, finance, industry, education, and policing, with strong national defense, and with opposition to abortion, affirmative action, gun control, and policies and laws that are viewed as challenging traditional social and family hierarchies and structure.
    """,
}

MODERATE_ROLE = {
    "name": "Moderate",
    "legend": "M",
    "role_refinement": """
    - An ideological category which designates a rejection of radical or extreme views, especially in regard to politics and religion. 
    - Someone occupying any mainstream position to avoid extreme views. 
    - Often described as politically unsophisticated, uninformed, or ideologically innocent, secretly partisan, ideologically cross-pressured, or extreme, with patterns of attitudes poorly described by a single ideological dimension
    """
}


# Load variables from .env file
load_dotenv()
# Prepare Logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def role_playing_check(role, claim, role_description, intent):
    return f"""Which text do you think the role would prefer? Return 'A' or 'B' as your answer and provide an explanation in JSON.
    Your task is to answer this question:
    How well does the Claim align with the Role's beliefs and intention?

    Details:
    - Role: {role}
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

def perform_gpt_metrics(args):
    # Argparse
    SAVE_PATH = args.save_path
    BASELINE_DF = args.baseline_df
    TEST_DF = args.test_df
    
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    # Save current args to output
    with open(f"{SAVE_PATH}/argsparse_run_config.json", 'w') as file:
        json.dump(vars(args), file)
        file.close()
        
    df = pd.read_json(TEST_DF)
    baseline_df = pd.read_json(BASELINE_DF)
    sampled_df = df.groupby('year', group_keys=False).sample(n=1, random_state=42)
    df = df[df['url'].isin(sampled_df['url'])]
    urls = list(set(df['url']))
    baseline_df = baseline_df[baseline_df['url'].isin(urls)]
    client = OpenAI(api_key=os.getenv("OPENAI_KEY"))
    prompts = []

    for _, row in df.iterrows():
        role_description = None
        if row['role'] == 'Democrat':
            role_description = DEMOCRAT_ROLE['role_refinement']
        elif row['role'] == 'Republican':
            role_description = REPUBLICAN_ROLE['role_refinement']
        elif row['role'] == 'Moderate':
            role_description = MODERATE_ROLE['role_refinement']
        else:
            raise ValueError()

        baseline_row = baseline_df[(baseline_df['url'] == row['url']) & (baseline_df['role_sequence'] == row['role_sequence']) & (baseline_df['round'] == row['round'])]
        round = row['round']
        prompts.append({'url': f"special_{round}_{row['role']}_{row['url']}_{row['role_sequence']}", "prompt": role_playing_check(row['role'], row['generated_claims']['Claim'], role_description, row['generated_claims']['Intent'])})
        prompts.append({'url': f"baseline_{round}_{row['role']}_{row['url']}_{row['role_sequence']}", "prompt": role_playing_check(row['role'], baseline_row['generated_claims'].iloc[0]['Claim'], role_description, baseline_row['generated_claims'].iloc[0]['Intent'],)})

    encoder = tiktoken.encoding_for_model('gpt-4o-mini')
    request_file_path = os.path.join(SAVE_PATH, f'role_playing_check.jsonl')

    with open(request_file_path, 'w') as outfile:
        for prompt in prompts:
            data = {
                "custom_id": prompt['url'],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "user", "content": prompt['prompt']},
                    ],
                }
            }
            json.dump(data, outfile)
            outfile.write('\n')
        outfile.close()
    
    batch_input_file = client.files.create(file=open(request_file_path, "rb"), purpose="batch")
    batch_input_file_id = batch_input_file.id
    output = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": f"Factuality Check for With and Without Role Definitions claims"
        }
    )

    with open(os.path.join(SAVE_PATH, 'output.json'), 'w') as outfile:
        detail = {
            "batch_id": output.id,
            "original_file": request_file_path
        }
        json.dump(detail, outfile)
        outfile.close()
    
if __name__ == '__main__':
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    parser = argparse.ArgumentParser(description='Politifact Link Extractor. Extract Personalities and Article Links from targeted URLs')
    parser.add_argument('--save_path', type=str, default=os.path.join('ablation_script_outputs', 'gpt_4o_mini_rp_eval'), help='Script output location')
    parser.add_argument('--baseline_df', type=str, default=os.path.join('script_outputs', 'role_playing_misinformation_labelling_output', '2025-03-21-00-48-13', 'role_playing_label_outputs.json'))
    parser.add_argument('--test_df', type=str, default=os.path.join('ablation_script_outputs', 'role_playing_misinformation_generation_output_without_role_definition', '2025-03-21-00-48-13', 'role_playing_outputs.json'))
    args = parser.parse_args()
    perform_gpt_metrics(args)