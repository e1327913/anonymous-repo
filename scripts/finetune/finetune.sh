# BERT Base
python ./src/finetune_encoders/finetune_bert.py --model_name="bert-base-uncased" --save_path="./script_outputs/finetuned_encoder_models/bert-base-uncased"

# BERT Large
python ./src/finetune_encoders/finetune_bert.py --model_name="bert-large-uncased" --save_path="./script_outputs/finetuned_encoder_models/bert-large-uncased"

# RoBERTa Base
python ./src/finetune_encoders/finetune_roberta.py --model_name="roberta-base" --save_path="./script_outputs/finetuned_encoder_models/roberta-base"

# RoBERTa Large
python ./src/finetune_encoders/finetune_roberta.py --model_name="roberta-large" --save_path="./script_outputs/finetuned_encoder_models/roberta-large"

# DeBERTa V3 Base
python ./src/finetune_encoders/finetune_deberta.py --model_name="microsoft/deberta-v3-base" --save_path="./script_outputs/finetuned_encoder_models/deberta-v3-base"

# DeBERTa V3 Large
python ./src/finetune_encoders/finetune_roberta.py --model_name="microsoft/deberta-v3-large" --save_path="./script_outputs/finetuned_encoder_models/deberta-v3-large" --train_batch_size=8 --dev_batch_size=8
