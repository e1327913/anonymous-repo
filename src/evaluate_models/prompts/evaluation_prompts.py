def zero_shot_evaluation_prompt(claim, fact_checking_evidences):
    all_fact_checking_evidences = ''
    for key, value in fact_checking_evidences.items():
        all_fact_checking_evidences += f'- {key}: {value} \n'
        
    prompt = f"""Based on the "Fact-Checking Evidence", select a "Label" from ["True", "Half-True", "False"] that is suitable for the "Claim" and provide an "Explanation".
    
Claim: <|claim|>

Fact-Checking Evidence:
<|fce|>

Output Format:
{{
    "Label": "<True / Half-True / False>",
    "Explanation": "<Short justification referencing your label choice>"
}}

"""
    
    prompt = prompt.replace('<|claim|>', claim).replace('<|fce|>', all_fact_checking_evidences)
    
    return prompt

def zero_shot_evaluation_prompt_cot(claim, fact_checking_evidences):
    all_fact_checking_evidences = ''
    for key, value in fact_checking_evidences.items():
        all_fact_checking_evidences += f'- {key}: {value} \n'
        
    prompt = f""" Consider the following in your evaluation:
Definitions:
- Claim:
    - A statement that state or assert that something is the case, typically without providing evidence or proof
- Fact-Checking Evidence: 
    - Verified information from reliable sources that is used to assess the veracity of a claim suspected of being misinformation. 
    - This can include statements from experts, official documentation, or trustworthy organizations that clarify misunderstandings or provide factual context to refute misleading claims.

Grading Scheme:
- The scheme has three ratings, in decreasing level of truthfulness
    - true : The statement is accurate and there’s nothing significant missing.
    - half-true : The statement is partially accurate but leaves out important details or takes things out of context.
    - false : The statement is not accurate.
    
Claim: <|claim|>

Fact-Checking Evidence:
<|fce|>

Your task is to assign an appropriate 'Label' to the 'Claim' based on its level of truthfulness, using the 'Grading Scheme'. 
To determine the claim's accuracy, you must fact-check it against the 'Fact-Checking Evidence'.
After assessing the claim, you must provide a detailed 'Explanation' justifying your choice of label. 
The final output must include both the 'Label' and 'Explanation' in JSON format.

Output Format:
{{
    "Label": "<True / Half-True / False>",
    "Explanation": "<Short justification referencing your label choice>"
}}

"""
    
    prompt = prompt.replace('<|claim|>', claim).replace('<|fce|>', all_fact_checking_evidences)
    
    return prompt
    
def few_shot_evaluation_prompt(claim, fact_checking_evidences):
    all_fact_checking_evidences = ''
    for key, value in fact_checking_evidences.items():
        all_fact_checking_evidences += f'- {key}: {value} \n'
        
    prompt = f"""Based on the "Fact-Checking Evidence", select a "Label" from ["True", "Half-True", "False"] that is suitable for the "Claim" and provide an "Explanation".

Examples:
Claim: "A proposed constitutional amendment “would allow anyone to run for a 3rd term. Including — Barack Obama.”
Fact-Checking Evidence:
- fce_source_0: "But the resolution doesn’t propose changing the 22nd Amendment so that former President Barack Obama — or any other president who served two consecutive terms — could run.",
- fce_source_1: "Ogles wants the amended amendment to say: 'No person shall be elected to the office of the president more than three times, nor be elected to any additional term after being elected to two consecutive terms, and no person who has held the office of the president, or acted as president, for more than two years of a term to which some other person was elected president shall be elected to the office of the president more than twice.'",
- fce_source_2: "That means former President Grover Cleveland, who died in 1908 and served two nonconsecutive presidential terms, would have been the only other former U.S. president eligible to run for reelection after serving two terms under the proposed amendment.",
- fce_source_3: "For the Constitution to be amended, Ogles’ bill would need to be approved by a two-thirds vote in both the House and Senate, and then ratified by three-fourths of the states."

This is the expected output format:
{{
    "Label": "False"
    "Explanation": "The claim that the proposed constitutional amendment “would allow anyone to run for a 3rd term, including Barack Obama” is false. The resolution introduced by U.S. Rep. Andy Ogles does not propose changes to the 22nd Amendment to allow presidents who have served two consecutive terms, like Barack Obama, to run for a third term. Instead, it specifically states that "no person shall be elected to the office of the president more than three times" but maintains restrictions for those who have already served two consecutive terms.
    The proposed amendment would only allow individuals who served nonconsecutive terms, such as former President Grover Cleveland, to run again. Furthermore, amending the Constitution requires an extensive process, including approval by two-thirds of both the House and Senate, as well as ratification by three-fourths of the states, making such a change highly unlikely.Therefore, the evidence provided directly contradicts the claim and clarifies the intent and scope of the proposed resolution."
}}

Claim: "Wisconsin makes it more difficult for its citizens to vote than almost any state in the nation. "
Fact-Checking Evidence:
- fce_source_0: "When asked to back up the claim, Common Cause Wisconsin Executive Director Jay Heck said he was pulling from the expertise of UW-Madison political science professor Barry Burden, who wrote previously for The Observatory that Wisconsin’s voter ID law is one of the strictest in the country. (Source: Barry Burden via The Observatory, April 16, 2024)",
- fce_source_1: "Burden repeated it in an email to PolitiFact Wisconsin, writing that 'Wisconsin demands more than nearly all of the other states' when it comes to getting a ballot. (Source: Email exchange with Barry Burden, UW-Madison)",
- fce_source_2: "The National Conference of State Legislatures lists Wisconsin as one of just nine states with 'strict' photo ID laws used to identify voters. (Source: National Conference of State Legislatures, accessed Jan. 18, 2025)",
- fce_source_3: "Besides the strict voter ID law, Wisconsin has stringent rules for voter registration drives, does not have automatic voter registration, does not have preregistration for young voters, does not allow all voters to join a permanent absentee ballot list, does not consider Election Day a public holiday and does require a witness’ signature on absentee ballots. (Source: National Conference of State Legislatures and Movement Advocacy Project, accessed Jan. 18, 2025)",
- fce_source_4: "Wisconsin ranks as the fifth-hardest state to vote in the country, according to the Cost of Voting Index. (Source: Cost of Voting Index, accessed Jan. 15, 2025)",
- fce_source_5: "In 1996, Wisconsin was ranked the fourth-most-accessible state in the nation for voters because it was among very few that offered same-day voter registration. (Source: Cost of Voting Index and Michael Pomante, accessed Jan. 15, 2025)",
- fce_source_6: "The state’s most dramatic drop occurred between 2011 and 2015, when former Republican Gov. Scott Walker signed the voter ID requirement into law and it took effect despite a swarm of lawsuits seeking to knock it down. (Source: CBS News, 'Walker signs photo ID requirement into law,' May 25, 2011)"

This is the expected output format:
{{
    "Label": "True"
    "Explanation": "The claim that Wisconsin makes it more difficult for its citizens to vote than almost any other state is supported by comprehensive evidence from credible sources. These include statements by political science expert Barry Burden, data from the National Conference of State Legislatures highlighting Wisconsin's strict voter ID laws, and findings from the Cost of Voting Index ranking Wisconsin as the fifth-hardest state for voting. Additional evidence points to the state's lack of measures like automatic voter registration, preregistration for young voters, and early voting, as well as its stringent requirements for absentee voting. Historical data also shows a significant decline in voter accessibility since 2011, when the voter ID law was enacted. Taken together, this evidence confirms that Wisconsin's voting laws and policies significantly hinder accessibility compared to most other states, making the claim accurate."
}}

Claim: <|claim|>
Fact-Checking Evidence:
<|fce|>

Please return this output only. This is the expected output format:
{{
    "Label": "<True / Half-True / False>",
    "Explanation": "<Short justification referencing your label choice>"
}}

    """
    
    prompt = prompt.replace('<|claim|>', claim).replace('<|fce|>', all_fact_checking_evidences)
    
    return prompt

def few_shot_evaluation_prompt_cot(claim, fact_checking_evidences):
    all_fact_checking_evidences = ''
    for key, value in fact_checking_evidences.items():
        all_fact_checking_evidences += f'- {key}: {value} \n'
        
    prompt = f""" Consider the following in your evaluation:
Definitions:
- Claim:
    - A statement that state or assert that something is the case, typically without providing evidence or proof
- Fact-Checking Evidence: 
    - Verified information from reliable sources that is used to assess the veracity of a claim suspected of being misinformation. 
    - This can include statements from experts, official documentation, or trustworthy organizations that clarify misunderstandings or provide factual context to refute misleading claims.

Grading Scheme:
- The scheme has three ratings, in decreasing level of truthfulness
    - true : The statement is accurate and there’s nothing significant missing.
    - half-true : The statement is partially accurate but leaves out important details or takes things out of context.
    - false : The statement is not accurate.

Your task is to assign an appropriate 'Label' to the 'Claim' based on its level of truthfulness, using the 'Grading Scheme'. 
To determine the claim's accuracy, you must fact-check it against the 'Fact-Checking Evidence'.
After assessing the claim, you must provide a detailed 'Explanation' justifying your choice of label. 
The final output must include both the 'Label' and 'Explanation' in JSON format.

Examples:
Claim: "A proposed constitutional amendment “would allow anyone to run for a 3rd term. Including — Barack Obama.”
Fact-Checking Evidence:
- fce_source_0: "But the resolution doesn’t propose changing the 22nd Amendment so that former President Barack Obama — or any other president who served two consecutive terms — could run.",
- fce_source_1: "Ogles wants the amended amendment to say: 'No person shall be elected to the office of the president more than three times, nor be elected to any additional term after being elected to two consecutive terms, and no person who has held the office of the president, or acted as president, for more than two years of a term to which some other person was elected president shall be elected to the office of the president more than twice.'",
- fce_source_2: "That means former President Grover Cleveland, who died in 1908 and served two nonconsecutive presidential terms, would have been the only other former U.S. president eligible to run for reelection after serving two terms under the proposed amendment.",
- fce_source_3: "For the Constitution to be amended, Ogles’ bill would need to be approved by a two-thirds vote in both the House and Senate, and then ratified by three-fourths of the states."

This is the expected output format:
{{
    "Label": "False"
    "Explanation": "The claim that the proposed constitutional amendment “would allow anyone to run for a 3rd term, including Barack Obama” is false. The resolution introduced by U.S. Rep. Andy Ogles does not propose changes to the 22nd Amendment to allow presidents who have served two consecutive terms, like Barack Obama, to run for a third term. Instead, it specifically states that "no person shall be elected to the office of the president more than three times" but maintains restrictions for those who have already served two consecutive terms.
    The proposed amendment would only allow individuals who served nonconsecutive terms, such as former President Grover Cleveland, to run again. Furthermore, amending the Constitution requires an extensive process, including approval by two-thirds of both the House and Senate, as well as ratification by three-fourths of the states, making such a change highly unlikely.Therefore, the evidence provided directly contradicts the claim and clarifies the intent and scope of the proposed resolution."
}}

Claim: "Wisconsin makes it more difficult for its citizens to vote than almost any state in the nation. "
Fact-Checking Evidence:
- fce_source_0: "When asked to back up the claim, Common Cause Wisconsin Executive Director Jay Heck said he was pulling from the expertise of UW-Madison political science professor Barry Burden, who wrote previously for The Observatory that Wisconsin’s voter ID law is one of the strictest in the country. (Source: Barry Burden via The Observatory, April 16, 2024)",
- fce_source_1:"Burden repeated it in an email to PolitiFact Wisconsin, writing that 'Wisconsin demands more than nearly all of the other states' when it comes to getting a ballot. (Source: Email exchange with Barry Burden, UW-Madison)",
- fce_source_2:"The National Conference of State Legislatures lists Wisconsin as one of just nine states with 'strict' photo ID laws used to identify voters. (Source: National Conference of State Legislatures, accessed Jan. 18, 2025)",
- fce_source_3:"Besides the strict voter ID law, Wisconsin has stringent rules for voter registration drives, does not have automatic voter registration, does not have preregistration for young voters, does not allow all voters to join a permanent absentee ballot list, does not consider Election Day a public holiday and does require a witness’ signature on absentee ballots. (Source: National Conference of State Legislatures and Movement Advocacy Project, accessed Jan. 18, 2025)",
- fce_source_4:"Wisconsin ranks as the fifth-hardest state to vote in the country, according to the Cost of Voting Index. (Source: Cost of Voting Index, accessed Jan. 15, 2025)",
- fce_source_5:"In 1996, Wisconsin was ranked the fourth-most-accessible state in the nation for voters because it was among very few that offered same-day voter registration. (Source: Cost of Voting Index and Michael Pomante, accessed Jan. 15, 2025)",
- fce_source_6:"The state’s most dramatic drop occurred between 2011 and 2015, when former Republican Gov. Scott Walker signed the voter ID requirement into law and it took effect despite a swarm of lawsuits seeking to knock it down. (Source: CBS News, 'Walker signs photo ID requirement into law,' May 25, 2011)"

This is the expected output format:
{{
    "Label": "True"
    "Explanation": "The claim that Wisconsin makes it more difficult for its citizens to vote than almost any other state is supported by comprehensive evidence from credible sources. These include statements by political science expert Barry Burden, data from the National Conference of State Legislatures highlighting Wisconsin's strict voter ID laws, and findings from the Cost of Voting Index ranking Wisconsin as the fifth-hardest state for voting. Additional evidence points to the state's lack of measures like automatic voter registration, preregistration for young voters, and early voting, as well as its stringent requirements for absentee voting. Historical data also shows a significant decline in voter accessibility since 2011, when the voter ID law was enacted. Taken together, this evidence confirms that Wisconsin's voting laws and policies significantly hinder accessibility compared to most other states, making the claim accurate."
}}

Claim: <|claim|>
Fact-Checking Evidence:
<|fce|>

Please return this output only. This is the expected output format:
{{
    "Label": "<True / Half-True / False>",
    "Explanation": "<Short justification referencing your label choice>"
}}

    """
    
    prompt = prompt.replace('<|claim|>', claim).replace('<|fce|>', all_fact_checking_evidences)
    
    return prompt