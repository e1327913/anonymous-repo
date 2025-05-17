# Llama 3.1-8B Instruct 

# Zero-Shot
# Round 1
echo "Zero-Shot Round 1"
python ./src/evaluate_models/evaluate_decoders.py --mode="zero_shot"  --save_path='./script_outputs/decoder_models_results/llama-3.1-8B-Instruct/zero_shot/round_1' --eval_data_path="./script_outputs/role_playing_misinformation_generated_datasets/round_1.json" --eval_batch_size=16
# Round 2
echo "Zero-Shot Round 2"
python ./src/evaluate_models/evaluate_decoders.py --mode="zero_shot"  --save_path='./script_outputs/decoder_models_results/llama-3.1-8B-Instruct/zero_shot/round_2' --eval_data_path="./script_outputs/role_playing_misinformation_generated_datasets/round_2.json" --eval_batch_size=16
# Round 3
echo "Zero-Shot Round 3"
python ./src/evaluate_models/evaluate_decoders.py --mode="zero_shot"  --save_path='./script_outputs/decoder_models_results/llama-3.1-8B-Instruct/zero_shot/round_3' --eval_data_path="./script_outputs/role_playing_misinformation_generated_datasets/round_3.json" --eval_batch_size=16
# Original
echo "Zero-Shot Original Round"
python ./src/evaluate_models/evaluate_decoders.py --mode="zero_shot"  --save_path='./script_outputs/decoder_models_results/llama-3.1-8B-Instruct/zero_shot/original' --eval_data_path="./script_outputs/role_playing_misinformation_generated_datasets/original.json" --eval_batch_size=16


# Zero-Shot CoT
# Round 1
echo "Zero-Shot CoT Round 1"
python ./src/evaluate_models/evaluate_decoders.py --mode="zero_shot_cot"  --save_path='./script_outputs/decoder_models_results/llama-3.1-8B-Instruct/zero_shot_cot/round_1' --eval_data_path="./script_outputs/role_playing_misinformation_generated_datasets/round_1.json" --eval_batch_size=16
# Round 2
echo "Zero-Shot CoT Round 2"
python ./src/evaluate_models/evaluate_decoders.py --mode="zero_shot_cot"  --save_path='./script_outputs/decoder_models_results/llama-3.1-8B-Instruct/zero_shot_cot/round_2' --eval_data_path="./script_outputs/role_playing_misinformation_generated_datasets/round_2.json" --eval_batch_size=16
# Round 3
echo "Zero-Shot CoT Round 3"
python ./src/evaluate_models/evaluate_decoders.py --mode="zero_shot_cot"  --save_path='./script_outputs/decoder_models_results/llama-3.1-8B-Instruct/zero_shot_cot/round_3' --eval_data_path="./script_outputs/role_playing_misinformation_generated_datasets/round_3.json" --eval_batch_size=16
# Original
echo "Zero-Shot CoT Original Round"
python ./src/evaluate_models/evaluate_decoders.py --mode="zero_shot_cot"  --save_path='./script_outputs/decoder_models_results/llama-3.1-8B-Instruct/zero_shot_cot/original' --eval_data_path="./script_outputs/role_playing_misinformation_generated_datasets/original.json" --eval_batch_size=16

# Few-Shot
# Round 1
echo "Few-Shot Round 1"
python ./src/evaluate_models/evaluate_decoders.py --mode="few_shot"  --save_path='./script_outputs/decoder_models_results/llama-3.1-8B-Instruct/few_shot/round_1' --eval_data_path="./script_outputs/role_playing_misinformation_generated_datasets/round_1.json" --eval_batch_size=16
# Round 2
echo "Few-Shot Round 2"
python ./src/evaluate_models/evaluate_decoders.py --mode="few_shot"  --save_path='./script_outputs/decoder_models_results/llama-3.1-8B-Instruct/few_shot/round_2' --eval_data_path="./script_outputs/role_playing_misinformation_generated_datasets/round_2.json" --eval_batch_size=16
# Round 3
echo "Few-Shot Round 3"
python ./src/evaluate_models/evaluate_decoders.py --mode="few_shot"  --save_path='./script_outputs/decoder_models_results/llama-3.1-8B-Instruct/few_shot/round_3' --eval_data_path="./script_outputs/role_playing_misinformation_generated_datasets/round_3.json" --eval_batch_size=16
# Original
echo "Few-Shot Original Round"
python ./src/evaluate_models/evaluate_decoders.py --mode="few_shot"  --save_path='./script_outputs/decoder_models_results/llama-3.1-8B-Instruct/few_shot/original' --eval_data_path="./script_outputs/role_playing_misinformation_generated_datasets/original.json" --eval_batch_size=16


# Few-Shot CoT
# Round 1
echo "Few-Shot CoT Round 1"
python ./src/evaluate_models/evaluate_decoders.py --mode="few_shot_cot"  --save_path='./script_outputs/decoder_models_results/llama-3.1-8B-Instruct/few_shot_cot/round_1' --eval_data_path="./script_outputs/role_playing_misinformation_generated_datasets/round_1.json" --eval_batch_size=16
# Round 2
echo "Few-Shot CoT Round 2"
python ./src/evaluate_models/evaluate_decoders.py --mode="few_shot_cot"  --save_path='./script_outputs/decoder_models_results/llama-3.1-8B-Instruct/few_shot_cot/round_2' --eval_data_path="./script_outputs/role_playing_misinformation_generated_datasets/round_2.json" --eval_batch_size=16
# Round 3
echo "Few-Shot CoT Round 3"
python ./src/evaluate_models/evaluate_decoders.py --mode="few_shot_cot"  --save_path='./script_outputs/decoder_models_results/llama-3.1-8B-Instruct/few_shot_cot/round_3' --eval_data_path="./script_outputs/role_playing_misinformation_generated_datasets/round_3.json" --eval_batch_size=16
# Original
echo "Few-Shot CoT Original Round"
python ./src/evaluate_models/evaluate_decoders.py --mode="few_shot_cot"  --save_path='./script_outputs/decoder_models_results/llama-3.1-8B-Instruct/few_shot_cot/original' --eval_data_path="./script_outputs/role_playing_misinformation_generated_datasets/original.json" --eval_batch_size=16
