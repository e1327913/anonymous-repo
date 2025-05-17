import os
import argparse
import json
import pandas as pd

def combine_dataframe(args):
    # Argparse
    SAVE_PATH = args.save_path
    ROUND_1_DF = args.round_1_df
    ROUND_2_DF = args.round_2_df
    ROUND_3_DF = args.round_3_df

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    # Save current args to output
    with open(f"{SAVE_PATH}/argsparse_config.json", 'w') as file:
        json.dump(vars(args), file)
        file.close()

    # Combine DF
    round_1_df = pd.read_json(ROUND_1_DF)
    round_2_df = pd.read_json(ROUND_2_DF)
    round_3_df = pd.read_json(ROUND_3_DF)

    df = pd.concat([round_1_df, round_2_df, round_3_df], ignore_index=True)
    df = df.sort_values(by='round')
    df.to_json(os.path.join(SAVE_PATH, 'role_playing_label_outputs.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-Round Persona Based Claim Generation')
    parser.add_argument('--save_path', type=str, default=os.path.join('script_outputs', 'role_playing_misinformation_labelling_output', '2025-03-21-00-48-13'), help='Output Location')
    parser.add_argument('--round_1_df', type=str, default=os.path.join('script_outputs', 'role_playing_misinformation_labelling_output_round_0', '2025-03-21-00-48-13', 'role_playing_label_outputs_round_0.json'), help='Dataset Location')
    parser.add_argument('--round_2_df', type=str, default=os.path.join('script_outputs', 'role_playing_misinformation_labelling_output_round_1', '2025-03-21-00-48-13', 'role_playing_label_outputs_round_1.json'), help='Dataset Location')
    parser.add_argument('--round_3_df', type=str, default=os.path.join('script_outputs', 'role_playing_misinformation_labelling_output_round_2', '2025-03-21-00-48-13', 'role_playing_label_outputs_round_2.json'), help='Dataset Location')
    args = parser.parse_args()
    combine_dataframe(args)