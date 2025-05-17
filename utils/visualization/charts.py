import os
import argparse
import logging
import pandas as pd
import matplotlib.pyplot as plt
import json
from dotenv import load_dotenv
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score, precision_score, recall_score, classification_report

# Load variables from .env file
load_dotenv()
# Prepare Logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

POLITIFACT_LABELS = {
    'true': 0,
    'half-true': 1,
    'false': 2,
}

def generate_horizontal_chart_type_A(result_df, save_path):
    dataset_order = ['round_3', 'round_2', 'round_1', 'original']

    # Pivot and reorder columns
    plot_df = result_df.pivot(index='model', columns='dataset_type', values='accuracy')
    plot_df = plot_df[plot_df.columns.intersection(dataset_order)]  # Keep only those in order
    plot_df = plot_df.reindex(columns=dataset_order)

    # Plot
    ax = plot_df.plot(kind='barh', figsize=(16, 13), width=0.9)

    plt.title('Accuracy by Model and Dataset Type')
    plt.ylabel('Model')
    plt.xlabel('Accuracy')
    plt.xticks(rotation=0)
    plt.legend(
        title='Dataset Type',
        loc='upper center',
        bbox_to_anchor=(0.5, -0.1),
        ncol=4
    )

    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f%%', label_type='edge', padding=5)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'horizontal_accuracy_chart_type_a.png'))
    plt.close()
    
def generate_horizontal_chart_type_B(result_df, save_path):
    dataset_order = ['round_3', 'round_2', 'round_1', 'original']

    # Pivot and reorder columns
    plot_df = result_df.pivot(index='dataset_type', columns='model', values='accuracy')
    plot_df = plot_df.reindex(index=dataset_order)

    # Plot
    ax = plot_df.plot(kind='barh', figsize=(16, 13), width=0.9)

    plt.title('Accuracy by Model and Dataset Type')
    plt.ylabel('Model')
    plt.xlabel('Accuracy')
    plt.xticks(rotation=0)
    plt.legend(
        title='Dataset Type',
        loc='upper center',
        bbox_to_anchor=(0.5, -0.1),
        ncol=4
    )

    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f%%', label_type='edge', padding=5)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'horizontal_accuracy_chart_type_b.png'))
    plt.close()
    
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

def generate_confusion_matrix_diagram(result_df, save_path):
    # Class labels
    display_labels = ['true', 'half-true', 'false']
    # label_map = {'true': 0, 'half-true': 1, 'false': 2}
    label_map = {0: 'True', 1: 'Half-True', 2: 'False'}

    models = sorted(result_df['model'].unique())
    dataset_types = ['original', 'round_1', 'round_2', 'round_3']

    for model in models:
        # Create a row of confusion matrices for this model
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        for j, dataset_type in enumerate(dataset_types):
            ax = axes[j // 2][j % 2]
            sub_df = result_df[(result_df['model'] == model) & (result_df['dataset_type'] == dataset_type)]

            row = sub_df.iloc[0]
            y_true = [label_map[x] for x in row['true']]
            y_pred = [label_map[x] for x in row['preds']]

            cm = confusion_matrix(y_true, y_pred, labels=list(label_map.values()))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(label_map.values()))
            disp.plot(ax=ax, cmap='Blues', colorbar=False)
            ax.set_title(f'{dataset_type.capitalize()}')
            if dataset_type == 'original':
                ax.set_xlabel('PolitiFact Label')
            else:
                ax.set_xlabel('Assigned Label (via Llama 3.1-8B Instruct)')
            ax.set_ylabel('Predicted Label')

        # Save the figure for this model
        model_path = os.path.join(save_path, 'confusion_matrix_rows')
        os.makedirs(model_path, exist_ok=True)
        save_file = os.path.join(model_path, f'{model}.png')
        plt.tight_layout()
        plt.savefig(save_file)
        plt.close(fig)

def calculate_metrics(result_df, save_path):
    # Class labels
    models = sorted(result_df['model'].unique())
    dataset_types = ['original', 'round_1', 'round_2', 'round_3']

    final_data = {
        'model': [],
        'dataset_type': [],
        'macro_f1': [],
        'micro_f1': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
    }
    report_arr = []
    for model in models:
        for dataset_type in dataset_types:
            sub_df = result_df[(result_df['model'] == model) & (result_df['dataset_type'] == dataset_type)]
            row = sub_df.iloc[0]
            y_true = row['true']
            y_pred = row['preds']
            macro_f1 = f1_score(y_true, y_pred, average='macro')
            micro_f1 = f1_score(y_true, y_pred, average='micro')
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='macro')
            recall = recall_score(y_true, y_pred, average='macro')
            print(f"{model} - {dataset_type} Metrics")
            print(f"Macro F1: {macro_f1:.2f}")
            print(f"Micro F1: {micro_f1:.2f}")
            print(f"Accuracy Score: {accuracy:.2f}")
            print(f"Precision: {precision:.2f}")
            print(f"Recall: {recall:.2f}")

            report_dict = classification_report(y_true, y_pred, output_dict=True)
            report_dict['model'] = model
            report_dict['dataset_type'] = dataset_type
            report_arr.append(report_dict)

            final_data['model'].append(model)
            final_data['dataset_type'].append(dataset_type)
            final_data['macro_f1'].append(macro_f1)
            final_data['micro_f1'].append(micro_f1)
            final_data['accuracy'].append(accuracy)
            final_data['precision'].append(precision)
            final_data['recall'].append(recall)
    
    df = pd.DataFrame(final_data)
    df = df.round(2)
    df.to_json(os.path.join(save_path, 'metrics.json'))

    df = pd.DataFrame(report_arr)
    df.to_json(os.path.join(save_path, 'classification_report.json'))


def generate_charts(args):
    ENCODER_RESULT_PATH = args.encoder_result_path
    DECODER_RESULT_PATH = args.decoder_result_path
    SAVE_PATH = args.save_path
    
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    # Save current args to output
    with open(f"{SAVE_PATH}/argsparse_config.json", 'w') as file:
        json.dump(vars(args), file)
        file.close()
    
    # Get all results
    result_df = pd.DataFrame({
        'model': [],
        'dataset_type': [],
        'accuracy': [],
        'preds': [],
        'true': []
    })
    
    # Get Encoder Results
    for model in os.listdir(ENCODER_RESULT_PATH):
        model_path = os.path.join(ENCODER_RESULT_PATH, model)
        if model == '.DS_Store':
            continue
        for dataset_type in os.listdir(model_path):
            result_path = os.path.join(model_path, dataset_type, 'eval.json')
            if dataset_type == '.DS_Store':
                continue
            df = pd.read_json(result_path)
            preds = []
            for result_label in df['eval_preds']:
                for label in result_label:
                    curr_preds = label.split(',')
                    curr_preds = [int(label) for label in curr_preds]
                    preds += curr_preds
            trues = []
            for result_label in df['eval_true']:
                for label in result_label:
                    curr_trues = label.split(',')
                    curr_trues = [int(label) for label in curr_trues]
                    trues += curr_trues
            row_df = pd.DataFrame({
                'model': model,
                'dataset_type': dataset_type,
                'accuracy': sum(pred == true for pred, true in zip(preds, trues)) / len(preds) * 100,
                'preds': [preds],
                'true': [trues],
                'explanation': None,
            })
            result_df = pd.concat([result_df, row_df])
    
    # Get GPT Results
    for model in os.listdir(DECODER_RESULT_PATH):
        model_path = os.path.join(DECODER_RESULT_PATH, model)
        if model == 'gpt-4o-mini':
            result_path = os.path.join(model_path, 'gpt_evaluation.json')
            gpt_df_groups = pd.read_json(result_path).get(['model', 'dataset_type', 'prompt_type', 'pred_label', 'true_label']).groupby(['dataset_type', 'prompt_type'])
            for _, data in gpt_df_groups:
                preds = [POLITIFACT_LABELS[label] for label in data['pred_label']]
                trues = [POLITIFACT_LABELS[label] for label in data['true_label']]
                model_names = list(set(data['model']))
                assert len(model_names) == 1
                prompt_type = list(set(data['prompt_type']))
                assert len(prompt_type) == 1
                dataset_type = list(set(data['dataset_type']))
                assert len(dataset_type) == 1
                model = model_names[0]
                prompt_type = prompt_type[0]
                dataset_type = dataset_type[0]
                row_df = pd.DataFrame({
                    'model': f"{model} ({prompt_type})",
                    'dataset_type': dataset_type,
                    'accuracy': sum(pred == true for pred, true in zip(preds, trues)) / len(preds) * 100,
                    'preds': [preds],
                    'true': [trues],
                    'explanation': None
                })
                result_df = pd.concat([result_df, row_df])
        elif model == 'llama-3.1-8B-Instruct':
            for prompt_type in os.listdir(model_path):
                prompt_path = os.path.join(model_path, prompt_type)
                for dataset_type in os.listdir(prompt_path):
                    if dataset_type == '.DS_Store':
                        continue
                    result_path = os.path.join(prompt_path, dataset_type, 'eval.json')
                    llama_df = pd.read_json(result_path)
                    llama_df_groups = llama_df.groupby('mode')
                    
                    for _, data in llama_df_groups:
                        preds = [POLITIFACT_LABELS[label.lower()] for label in data['pred_label']]
                        trues = [POLITIFACT_LABELS[label] for label in data['label']]
                        row_df = pd.DataFrame({
                            'model': f"{model} ({prompt_type})",
                            'dataset_type': dataset_type,
                            'accuracy': sum(pred == true for pred, true in zip(preds, trues)) / len(preds) * 100,
                            'preds': [preds],
                            'true': [trues],
                            'explanation': [data['pred_label_explanation']]
                        })
                        result_df = pd.concat([result_df, row_df])
        
    result_df = result_df.reset_index()
    result_df.to_json(os.path.join(SAVE_PATH, 'final_results.json'))

    # Generate Charts
    generate_horizontal_chart_type_A(result_df, SAVE_PATH)
    generate_horizontal_chart_type_B(result_df, SAVE_PATH)
    
    # Generate Confusion Matrix
    generate_confusion_matrix_diagram(result_df, SAVE_PATH)

    # Store all results in an excel sheet
    calculate_metrics(result_df, SAVE_PATH)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-Round Persona Based Claim Generation')
    parser.add_argument('--encoder_result_path', default=os.path.join('script_outputs', 'finetuned_encoder_models_results'))
    parser.add_argument('--decoder_result_path', default=os.path.join('script_outputs', 'decoder_models_results'))
    parser.add_argument('--save_path', default=os.path.join('script_outputs', 'visualization'))
    args = parser.parse_args()
    generate_charts(args)