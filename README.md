# Multi-Round Persona-Conditioned Claim Generation: Modeling the Evolution of Misinformation with LLMs (EMNLP 2025 Submission)

## Abstract
Misinformation evolves as it spreads, shifting in language, framing, and moral emphasis to adapt to new audiences. However, current misinformation detection approaches implicitly assume that misinformation is static. We introduce a multi-round, persona-conditioned framework that simulates how claims are iteratively reinterpreted by agents with distinct ideological perspectives. Our approach uses an uncensored large language model (LLM) to generate persona-specific claims across multiple rounds, conditioning each generation on outputs from the previous round, enabling the study of misinformation evolution. We evaluate the generated claims through human and LLM-based annotations, cognitive effort metrics (readability, perplexity), emotion evocation metrics (sentiment analysis, morality), clustering, and downstream classification. Results show that ideological and stylistic shifts reduce the macro-F1 scores of commonly used misinformation detectors by up to 0.36, despite preserving topic coherence.

## 🚀 How to Run

### 1. Environment Setup
Create an environment with Python 3.10.

Install dependencies with pip:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
pip install numpy==1.24.4
pip install transformers==4.31.0
```

### 2. Dataset 
1. Scrape articles from PolitiFact. 
```bash
python ./utils/politifact/politifact_link_extractor.py
```

2. Extract all of the information from the extracted HTMLs.
```bash
python ./utils/politifact/politifact_articles_extractor.py
```

3. Format all the extracted details
```bash
python ./utils/politifact/politifact_articles_formatter.py
```

4. Annotation request to GPT-4o-mini
```bash
python ./utils/politifact/politifact_extractor_openai_batch_request.py
```

5. Retrieve annotation output from GPT-4o-mini.
```bash
python ./utils/politifact/politifact_extractor_openai_batch_check.py
```

6. Format all the annotation output from GPT-4o-mini
```bash
python ./utils/politifact/format_batch_outputs.py
```

7. Prepare Train, Dev and Test dataset 
```bash
python ./utils/dataset_formatting/train_dev_test.py
```

## Framework
### 1. Misinformation Generation
```bash
python ./role_playing_misinformation_generation_cuda.py
```

### 2. Misinformation Labeling
```bash
python ./role_playing_misinformation_labelling_cuda.py
```

## Prepare Generated Datasets
```bash
python ./utils/generate_dataset/classification_test_dataset.py
```

## Classification 
### 1. Finetuning Encoder Models
```bash
bash ./scripts/finetune/finetune.sh
```

### 2. Testing Encoder and Decoder Models
```bash
bash ./scripts/evaluate/encoder.sh
bash ./scripts/evaluate/decoder.sh
bash ./scripts/evaluate/gpt.sh
```

### 3. Visualize
```bash
python ./utils/visualization/charts.py
```
