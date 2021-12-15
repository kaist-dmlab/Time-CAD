#%%
import os

import pandas as pd
import numpy as np
import copy
from tqdm import tqdm
from plot import plot

from utils.evaluator import evaluate, set_thresholds
from utils.evaluator_seg import compute_anomaly_scores, compute_metrics
# Univariate
from utils.data_loader import load_kpi, load_IoT_fridge
# Multivariate
from utils.data_loader import load_samsung, load_energy, load_unsw, load_IoT_modbus

def _elements(array):
    return array.ndim and array.size

def train(AE_model, Temporal_AE_model, model_name, window_size, stride, lamda_t, wavelet_num, seed, dataset, temporal=False, decomposition=False, segmentation=False):
    ts_scores = {'dataset': [], 'f1': [], 'precision': [], 'recall': [], 'pr_auc': [], 'roc_auc': [], 'th_index': [], 'predicted_index': []}
    seg_scores = {'dataset': [], 'avg_f1': [], 'avg_p': [], 'avg_r': [], 'max_p': [], 'max_r': [],  'max_f1': [], 'correct_count': [], 'correct_ratio': []}

    if temporal == True:
        datasets_auxiliary = globals()[f'load_{dataset}'](window_size, stride, lamda_t, wavelet_num, temporal=temporal)
        ax_trains, ax_tests = datasets_auxiliary['x_train'], datasets_auxiliary['x_test']
    
    # There are eight cases #1-1~#1-4 & #2-1~#2-4
    # 1) decomposition==True: Decompose time series and evaluate through traditional metrics (Temporal)
    # 4) decomposition==False: Evaluate through traditional metrics with common methods
    if segmentation == False:
        datasets = globals()[f'load_{dataset}'](window_size, stride, lamda_t, wavelet_num, decomposition=decomposition, segmentation=segmentation)
        x_trains, x_tests, y_tests = datasets['x_train'], datasets['x_test'], datasets['y_test']
        test_seq, label_seq = datasets['test_seq'], datasets['label_seq']
        if decomposition == True:
            train_residual, test_residual = datasets['x_train_resid'], datasets['x_test_resid']
        
        per_window_idx = []
        for data_num in tqdm(range(len(x_trains))):
            # 1) if decomposition == True
            if decomposition == True:
                X_test = x_tests[data_num]
                residual_X_train = train_residual[data_num]
                residual_X_test = test_residual[data_num]

                # 1-1) temporal=True, decomposition=True, Segmentation=False
                if temporal == True:
                    X_train_ax = ax_trains[data_num]
                    X_test_ax = ax_tests[data_num]                    
                    model = Temporal_AE_model(X_train_ax, residual_X_train)
                    rec_x = model.predict([X_test_ax, residual_X_test])
                    thresholds = set_thresholds(residual_X_test, rec_x, is_reconstructed=True)
                    test_scores = evaluate(thresholds, residual_X_test, rec_x, y_tests[data_num], is_reconstructed=True)
                # 2-1) temporal=False, decomposition=True, Segmentation=False
                else:
                    if model_name == "MS-RNN":
                        model = AE_model(residual_X_train)
                        rec_x = [np.flip(rec, axis=1) for rec in model.predict(residual_X_test)]
                        thresholds = set_thresholds(residual_X_test, rec_x, is_reconstructed=True, scoring='square_median')
                        test_scores = evaluate(thresholds, residual_X_test, rec_x, y_tests[data_num], is_reconstructed=True, scoring='square_median')
                    else:
                        model = AE_model(residual_X_train)
                        rec_x = model.predict(residual_X_test)
                        thresholds = set_thresholds(residual_X_test, rec_x, is_reconstructed=True)
                        test_scores = evaluate(thresholds, residual_X_test, rec_x, y_tests[data_num], is_reconstructed=True)

            # 4) if decomposition == False
            else:
                X_train = x_trains[data_num]
                X_test = x_tests[data_num]              
                
                # 1-4) temporal=True, decomposition=False, segmentation=False
                if temporal == True:
                    X_train_ax = ax_trains[data_num]
                    X_test_ax = ax_tests[data_num]                    
                    model = Temporal_AE_model(X_train_ax, X_train)
                    rec_x = model.predict([X_test_ax, X_test])
                    thresholds = set_thresholds(X_test, rec_x, is_reconstructed=True)
                    test_scores = evaluate(thresholds, X_test, rec_x, y_tests[data_num], is_reconstructed=True)                  
                # 2-4) temporal=False, decomposition=False, segmentation:False
                else:
                    if model_name == "MS-RNN":
                        model = AE_model(X_train)
                        rec_x = [np.flip(rec, axis=1) for rec in model.predict(X_test)]
                        thresholds = set_thresholds(X_test, rec_x, is_reconstructed=True, scoring='square_median')
                        test_scores = evaluate(thresholds, X_test, rec_x, y_tests[data_num], is_reconstructed=True, scoring='square_median')
                    else:
                        model = AE_model(X_train)
                        rec_x = model.predict(X_test)
                        thresholds = set_thresholds(X_test, rec_x, is_reconstructed=True)
                        test_scores = evaluate(thresholds, X_test, rec_x, y_tests[data_num], is_reconstructed=True)

            ts_scores['dataset'].append(f'Data{data_num+1}')
            ts_scores['f1'].append(np.max(test_scores['f1']))
            ts_scores['precision'].append(np.mean(test_scores['precision']))
            ts_scores['recall'].append(np.mean(test_scores['recall']))
            ts_scores['pr_auc'].append(test_scores['pr_auc'])
            ts_scores['roc_auc'].append(test_scores['roc_auc'])
            th_index = int(np.median(np.where(test_scores['f1']==np.max(test_scores['f1']))[0]))
            ts_scores['th_index'].append(th_index)
            print(f'{seed}th {model_name} Data{data_num+1}', np.max(test_scores['f1']), np.mean(test_scores['precision']), np.mean(test_scores['recall']), test_scores['pr_auc'], test_scores['roc_auc'])
            
            pred_anomal_idx = []
            for t in range(len(X_test)):
                pred_anomalies = np.where(test_scores['rec_errors'][t] > thresholds[th_index])[0]
                isEmpty = (_elements(pred_anomalies) == 0)
                if isEmpty:
                    pass
                else:
                    if pred_anomalies[0] == 0:
                        pred_anomal_idx.append(t)
            per_window_idx.append(pred_anomal_idx)
        ts_scores['predicted_index'].extend(per_window_idx)

        scores_all = copy.deepcopy(ts_scores)
        del ts_scores['th_index']
        results_df = pd.DataFrame(ts_scores)
        print("@"*5, f'{seed}th Seed {model_name} R{decomposition}_T{temporal}_Ts', "@"*5)
        print(results_df.groupby('dataset').mean())
        
        save_results_path = f'./results/{dataset}/Ts'
        try:
            if not(os.path.isdir(save_results_path)):
                os.makedirs(os.path.join(save_results_path), exist_ok=True)
        except OSError as e:
            print("Failed to create directory!!!!!")

        results_df.to_csv(f'{save_results_path}/{model_name}_R{decomposition}_T{temporal}_ts_seed{seed}.csv', index=False)
        plot(model_name, ts_scores, test_seq, label_seq, seed, save_results_path, decomposition, temporal)

    # 2) decomposition==True: Decompose time series and evalutate new metrics (Temporal+Seg_evaluation)
    # 3) decomposition==False: Evaluate through new metrics with common methods (Seg_evaluation)
    elif segmentation == True:
        datasets = globals()[f'load_{dataset}'](window_size, stride, lamda_t, wavelet_num, decomposition=decomposition, segmentation=segmentation)
        x_trains, x_tests = datasets['x_train'], datasets['x_test']
        y_tests, y_segment_tests = datasets['y_test'], datasets['y_segment_test']
        if decomposition == True:
            train_residual, test_residual = datasets['x_train_resid'], datasets['x_test_resid']
        
        per_window_idx = []
        for data_num in tqdm(range(len(x_trains))):
            # 2) if decomposition == True
            if decomposition == True:
                residual_X_train = train_residual[data_num]
                residual_X_test = test_residual[data_num]

                # 1-2) temporal=True, decomposition=True, segmentation=True
                if temporal == True:
                    X_train_ax = ax_trains[data_num]
                    X_test_ax = ax_tests[data_num]                    
                    model = Temporal_AE_model(X_train_ax, residual_X_train)
                    scores = compute_anomaly_scores(residual_X_test, model.predict([X_test_ax, residual_X_test]))
                    test_scores = compute_metrics(scores, y_tests[data_num], y_segment_tests[data_num])
                else:
                # 2-2) temporal=False, decomposition=True, segmentation=True
                    if model_name == "MS-RNN":
                        model = AE_model(residual_X_train)
                        rec_x = np.mean([np.flip(rec, axis=1) for rec in model.predict(residual_X_test)], axis=0)
                        scores = compute_anomaly_scores(residual_X_test, rec_x, scoring='square_median')
                        test_scores = compute_metrics(scores, y_tests[data_num], y_segment_tests[data_num])
                    else:                
                        model = AE_model(residual_X_train)
                        scores = compute_anomaly_scores(residual_X_test, model.predict(residual_X_test))
                        test_scores = compute_metrics(scores, y_tests[data_num], y_segment_tests[data_num])
            
            # 3) if decomposition == False
            else:
                X_train = x_trains[data_num]
                X_test = x_tests[data_num]

                # 1-3) temporal=True, decomposition=False, segmentation=True
                if temporal == True:
                    X_train_ax = ax_trains[data_num]
                    X_test_ax = ax_tests[data_num]                     
                    model = Temporal_AE_model(X_train_ax, X_train)
                    scores = compute_anomaly_scores(X_test, model.predict([X_test_ax, X_test]))
                    test_scores = compute_metrics(scores, y_tests[data_num], y_segment_tests[data_num])
                # 2-3) temporal=False, decomposition=False, segmentation=True
                else:
                    if model_name == "MS-RNN":
                        model = AE_model(X_train)
                        rec_x = np.mean([np.flip(rec, axis=1) for rec in model.predict(X_test)], axis=0)
                        scores = compute_anomaly_scores(X_test, rec_x, scoring='square_median')
                        test_scores = compute_metrics(scores, y_tests[data_num], y_segment_tests[data_num])
                    else:                    
                        model = AE_model(X_train)
                        scores = compute_anomaly_scores(X_test, model.predict(X_test))
                        test_scores = compute_metrics(scores, y_tests[data_num], y_segment_tests[data_num])             

            seg_scores['dataset'].append(f'Data{data_num+1}')
            seg_scores['max_f1'].append(np.max(test_scores['f1']))
            seg_scores['max_p'].append(np.max(test_scores['precision']))
            seg_scores['max_r'].append(np.max(test_scores['recall']))  
            seg_scores['avg_f1'].append(np.average(test_scores['f1']))
            seg_scores['avg_p'].append(np.average(test_scores['precision']))
            seg_scores['avg_r'].append(np.average(test_scores['recall']))
            seg_scores['correct_count'].append(np.average(test_scores['count']))
            seg_scores['correct_ratio'].append(np.average(test_scores['ratio']))

            print(f'{seed}th {model_name} Data{data_num+1}', np.max(test_scores['f1']), np.mean(test_scores['precision']), np.mean(test_scores['recall']), np.mean(test_scores['count']), np.mean(test_scores['ratio']))
        
        results_df = pd.DataFrame(seg_scores)
        print("@"*5, f'{seed}th Seed {model_name} R{decomposition}_T{temporal}_Seg', "@"*5)
        print(results_df.groupby('dataset').mean())

        save_results_path = f'./results/{dataset}/Seg'
        try:
            if not(os.path.isdir(save_results_path)):
                os.makedirs(os.path.join(save_results_path), exist_ok=True)
        except OSError as e:
            print("Failed to create directory!!!!!")

        results_df.to_csv(f'{save_results_path}/{model_name}_R{decomposition}_T{temporal}_seg_seed{seed}.csv', index=False)
# %%
