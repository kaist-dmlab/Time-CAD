import os, json
import numpy as np
import pandas as pd

from data_loader import *

def preprocess_uploaded_file(filepath):
    if filepath.split('.')[-1] in ['xls', 'xlsx']:
        data = pd.read_excel(filepath)
        chart_data = {
            'date': '2022-01-01',
            'var1': 100,
            'var2': 200,
            'var3': 300,
            'score': 1.5,
            'label': 0
        }
        os.remove(filepath)
        return {'status': 200, 'data': chart_data}
    elif filepath.split('.')[-1] == 'csv':
        df = pd.read_csv(filepath)
        
        df['anomaly_scores'] = np.random.random(size=df.shape[0])
        
        chart_data = json.loads(df.to_json(orient='records'))
        os.remove(filepath)
        return {'status': 200, 'data': chart_data, 'columns': df.columns.tolist()[1:]}
    else:
        return {'status': 400, 'message': 'upsupported file type'}


def preprocess_data(test_df, seq_length, stride, weight, wavelet_num, historical=False, temporal=False, decomposition=False, segmentation=False):
    # interval: 10 min
    # seq. length: 36:1 (i.e., 6 hour)
    # source: SAMSUNG

    x_train, x_test, y_test = [], [], []
    y_segment_test = []
    x_train_resid, x_test_resid = [], []
    label_seq, test_seq = [], []

    if temporal == True:
        # train_df = np.array(add_temporal_info('samsung', train_df, train_df.date))
        # train_df = train_df[:, 6:-1].astype(float)
        test_df = np.array(add_temporal_info('samsung', test_df, test_df.date))
        test_df = test_df[:, 6:-1].astype(float)
    else:
        if decomposition == True:
            # train_holiday = np.array(add_temporal_info('samsung', train_df, train_df.date)['holiday'])
            test_holiday = np.array(add_temporal_info(
                'samsung', test_df, test_df.date)['holiday'])
            # train_weekend = np.array(add_temporal_info('samsung', train_df, train_df.date)['is_weekend'])
            test_weekend = np.array(add_temporal_info(
                'samsung', test_df, test_df.date)['is_weekend'])
            # train_temporal = (train_holiday + train_weekend).reshape(-1, 1)
            test_temporal = (test_holiday + test_weekend).reshape(-1, 1)
        # train_df = np.array(train_df)
        # train_df = train_df[:, 1:-1].astype(float)
        test_df = np.array(test_df)
        labels = test_df[:, -1].astype(int)
        test_df = test_df[:, 1:-1].astype(float)

        scaler = MinMaxScaler()
        # train_df = scaler.fit_transform(train_df)
        test_df = scaler.fit_transform(test_df)

    if decomposition == True:
        stl_loader = load_STL_results(train_df, test_df)

        # train_seasonal = stl_loader['train_seasonal'] 
        test_seasonal = stl_loader['test_seasonal']
        # train_trend = stl_loader['train_trend'] 
        test_trend = stl_loader['test_trend']
        # train_normal = train_seasonal + train_trend
        test_normal = test_seasonal + test_trend
        # x_train_normal = _create_sequences(train_normal, seq_length, stride, historical)
        x_test_normal = _create_sequences(
            test_normal, seq_length, stride, historical)

        print("#"*10, "Deep Decomposer Generating...", "#"*10)
        deep_pattern = decompose_model(x_test_normal)
        # deep_train = deep_pattern['rec_train']
        deep_test = deep_pattern['rec_test']
        # deep_train_pattern = _decreate_sequences(deep_train)
        deep_test_pattern = _decreate_sequences(deep_test)

        # train_resid = (train_df - deep_train_pattern) * (1 + weight * train_temporal)
        test_resid = (test_df - deep_test_pattern) * \
            (1 + weight * test_temporal)

        # Wavelet transformation
        # train_resid_wav = _wavelet(train_resid)
        test_resid_wav = _wavelet(test_resid)

        # train_resid_wavelet = _wavelet(train_resid_wav)
        test_resid_wavelet = _wavelet(test_resid_wav)

        for iter in range(wavelet_num):
            # train_resid_wavelet = _wavelet(train_resid_wavelet)
            test_resid_wavelet = _wavelet(test_resid_wavelet)

    if temporal == True:
        if seq_length > 0:
            # x_train.append(_create_sequences(train_df, seq_length, stride, historical))
            x_test.append(_create_sequences(
                test_df, seq_length, stride, historical))
        else:
            # x_train.append(train_df)
            x_test.append(test_df)
    else:
        if seq_length > 0:
            # x_train.append(_create_sequences(train_df, seq_length, stride, historical))
            x_test.append(_create_sequences(
                test_df, seq_length, stride, historical))
            y_test.append(_create_sequences(
                labels, seq_length, stride, historical))
        else:
            # x_train.append(train_df)
            x_test.append(test_df)
            y_test.append(labels)

        if decomposition == True:
            # x_train_resid.append(_create_sequences(train_resid_wavelet, seq_length, stride, historical))
            x_test_resid.append(_create_sequences(
                test_resid_wavelet, seq_length, stride, historical))

        y_segment_test.append(_count_anomaly_segments(labels)[1])
        label_seq.append(labels)

        # For plot traffic raw data
        test_seq.append(test_df)

    # Only return temporal auxiliary information
    if temporal == True:
        return {'x_test': x_test}

    # There are four cases.
    # 1) Decompose time series and evaluate through traditional metrics
    if (decomposition == True) and (segmentation == False):
        return {'x_test': x_test, 'y_test': y_test,
                'x_test_resid': x_test_resid,
                'label_seq': label_seq, 'test_seq': test_seq}
    # 2) Decompose time series and evalutate new metrics
    elif (decomposition == True) and (segmentation == True):
        return {'x_test': x_test,
                'y_test': label_seq, 'y_segment_test': y_segment_test,
                'x_test_resid': x_test_resid}
    # 3) Evaluate through new metrics with common methods
    elif (decomposition == False) and (segmentation == True):
        return {'x_test': x_test,
                'y_test': label_seq, 'y_segment_test': y_segment_test}
    # 4) Evaluate through traditional metrics with common methods
    elif (decomposition == False) and (segmentation == False):
        return {'x_test': x_test, 'y_test': y_test,
                'label_seq': label_seq, 'test_seq': test_seq}
