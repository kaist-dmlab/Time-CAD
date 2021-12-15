import os

import numpy as np
import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt

def plot(model_name, ts_scores, test_seq, label_seq, seed, save_results_path, decomposition, temporal):
    save_fig_path = f'{save_results_path}/plot'

    # Plot traffic dataset and labels
    for data_num, t in enumerate(test_seq):
        sns.set_palette("Pastel1", 8)
        label_index = np.where(label_seq[data_num]==1)[0]
        predict_label_index = ts_scores['predicted_index'][data_num]
        pd.DataFrame(t).plot(figsize=(15, 10), label=None)
        max_value = np.max(np.max(t)) + 0.1
        min_value = np.min(np.min(t)) - 0.1
        median_value = (max_value + min_value) / 2
        plt.vlines(label_index, median_value+1, max_value+0.25, color = 'red', linewidth=1, linestyles='solid', alpha=0.5, label='Ground Truth')
        plt.vlines(predict_label_index, min_value-0.25, median_value-1, color = 'blue', linewidth=1, linestyles='solid', alpha=0.5, label='Predicted')
        plt.legend(loc='upper center', ncol=10, bbox_to_anchor=(0.5, 0.999))
        plt.title(f"{model_name} Model Data{data_num+1}")
        # plt.show()

        try:
            if not(os.path.isdir(save_fig_path)):
                os.makedirs(os.path.join(save_fig_path), exist_ok=True)
        except OSError as e:
            print("Failed to create directory!!!!!")

        plt.savefig(f'{save_fig_path}/{model_name}_Data{data_num+1}_R{decomposition}_T{temporal}_seed{seed}.png')

    # Plot only labels
    for data_num, t in enumerate(test_seq):
        save_fig_path = f'{save_results_path}/label_plot'

        label_index = np.where(label_seq[data_num]==1)[0]
        predict_label_index = ts_scores['predicted_index'][data_num]
        max_value = np.max(np.max(t)) + 0.1
        min_value = np.min(np.min(t)) - 0.1
        median_value = (max_value + min_value) / 2
        plt.figure(figsize=(15, 3))
        plt.vlines(label_index, median_value+0.1, max_value, color = 'red', linewidth=1, linestyles='solid', label='Ground Truth')
        plt.vlines(predict_label_index, min_value, median_value-0.1, color = 'blue', linewidth=1, linestyles='solid', label='Predicted')
        plt.legend(loc='upper center', ncol=10, bbox_to_anchor=(0.5, 1.25))
        plt.title(f"{model_name} Model Data{data_num+1}")
        # plt.show()

        try:
            if not(os.path.isdir(save_fig_path)):
                os.makedirs(os.path.join(save_fig_path), exist_ok=True)
        except OSError as e:
            print("Failed to create directory!!!!!")

        plt.savefig(f'{save_fig_path}/Label-{model_name}_Data{data_num+1}_R{decomposition}_T{temporal}_seed{seed}.png')
