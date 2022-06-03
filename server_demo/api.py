from math import floor, ceil
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from server_demo.firestore import Firestore


def stats(length: int):
    """
    :param length: the length of timestamps to calculate percent changes
    :return: the count and percent change of anomalies
    """
    firestore = Firestore()
    df = firestore.get_full_data()
    current_df = df.tail(length)
    past_df = df.iloc[-2 * length:-length]
    var_columns = [c for c in df.columns if (not c.startswith('label') and not c.startswith('score') and c not in ['date'])]
    result = {var: df[f'label_{var}'].sum() for var in var_columns}

    return {
        'total': {
            'n': int(current_df['label'].sum()),
            'percent': float((current_df['label'].sum() - past_df['label'].sum()) / past_df['label'].sum() * 100) if past_df['label'].sum() else 0
        },
        'variable': [
            {'name': k, 'n': int(v), 'percent': float((current_df[f'label_{k}'].sum() - past_df[f'label_{k}'].sum()) / past_df[f'label_{k}'].sum() * 100) if past_df[f'label_{k}'].sum() else 0} for k, v in result.items()
        ]
    }


def hourly_chart(length: int):
    """
    :param length: length of data in timestamps to calculate
    :return: the hourly count of anomalies
    """
    firestore = Firestore()
    df = firestore.get_full_data()
    df = df.tail(length)
    df = df[df['label'] == 1]
    df['date'] = pd.to_datetime(df['date'])
    df['hour'] = df['date'].dt.hour
    result = df['hour'].value_counts()
    result = [{'time': f'{k}:00', 'value': v} for k, v in result.to_dict().items()]
    return result


def weekly_chart(length: int):
    """
    :param length: length of data in timestamps to calculate
    :return: the weekly count of anomalies
    """
    firestore = Firestore()
    df = firestore.get_full_data()
    df = df.tail(length)
    df = df[df['label'] == 1]
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.day_name()
    result = df['day_of_week'].value_counts()
    result = [{'time': k, 'value': v} for k, v in result.to_dict().items()]
    print(result)
    return result


def full_stats(length: int):
    variable_stats = stats(length)
    hourly_stats = hourly_chart(length)
    weekly_stats = weekly_chart(length)
    data = {
        'stats': variable_stats,
        'hourly': hourly_stats,
        'weekly': weekly_stats
    }
    print(data)
    firestore = Firestore()
    firestore.add_stats(data)


def close_pattern_chart(variable_name: str, anomaly_timestamp: str, interval: int, count: int) -> List[dict]:
    """
    :param variable_name: name of the variable to search for
    :param anomaly_timestamp: timestamp of the anomaly to find similar patterns for
    :param interval: length of each pattern found
    :param count: the number of similar patterns to find
    :return: an array of similar patterns as a list of rows
    """
    firestore = Firestore()
    df = firestore.get_full_data()
    df = df[['date', variable_name, f'label_{variable_name}']]
    print(df)

    anomaly_idx = df.index[df['date'] == anomaly_timestamp][0]
    anomaly_range = [anomaly_idx - floor((interval - 1) / 2), anomaly_idx + ceil((interval - 1) / 2)]
    if anomaly_range[0] < 0:
        anomaly_range[1] = anomaly_range[1] - anomaly_range[0]
        anomaly_range[0] = max(0, anomaly_range[0])
    elif anomaly_range[1] >= len(df):
        anomaly_range[0] = anomaly_range[0] - (anomaly_range[1] - len(df) + 1)
        anomaly_range[1] = min(anomaly_range[1], len(df) - 1)
    anomaly = df.loc[anomaly_range[0]:anomaly_range[1], variable_name]
    print(anomaly)
    df['corr'] = df[variable_name].rolling(interval).apply(lambda r: pearsonr(r, anomaly)[0])
    order = np.argsort(df['corr'].tolist())[::-1]
    order = order[interval:interval + count]
    print(order)

    result = []
    for idx in order:
        match = df[idx - interval + 1: idx + 1].copy()
        match['range'] = f'{match["date"].iloc[0]}-{match["date"].iloc[-1]}'
        match['name'] = variable_name
        match.rename(columns={variable_name: 'value', f'label_{variable_name}': 'label'}, inplace=True)
        match.drop('corr', axis=1, inplace=True)
        print(match)
        data = match.to_dict('records')
        print(data)
        result.extend(data)
    print(result)
    return result


def past_close_patterns(anomaly_timestamp: str, interval: int) -> List[dict]:
    """
    :param anomaly_timestamp: timestamp of the anomaly to find close patterns for
    :param interval: length of each pattern found
    :return: a list of points in the close pattern
    """
    firestore = Firestore()
    df = firestore.get_full_data().head(100)

    anomaly_idx = df.index[df['date'] == anomaly_timestamp][0]
    anomaly_range = [anomaly_idx - floor((interval - 1) / 2), anomaly_idx + ceil((interval - 1) / 2)]
    if anomaly_range[0] < 0:
        anomaly_range[1] = anomaly_range[1] - anomaly_range[0]
        anomaly_range[0] = max(0, anomaly_range[0])
    elif anomaly_range[1] >= len(df):
        anomaly_range[0] = anomaly_range[0] - (anomaly_range[1] - len(df) + 1)
        anomaly_range[1] = min(anomaly_range[1], len(df) - 1)
    var_columns = [c for c in df.columns if (not c.startswith('label') and not c.startswith('score') and c not in ['date'])]
    anomaly = df.loc[anomaly_range[0]:anomaly_range[1], var_columns]
    print('anomaly', anomaly)

    for var in var_columns:
        df[f'corr_{var}'] = df[var].rolling(interval).apply(lambda r: pearsonr(r, anomaly[var])[0])
    # correlation is the mean of the correlations for each variable
    df['corr'] = df[[f'corr_{var}' for var in var_columns]].mean(axis=1)

    order = np.argsort(df['corr'].tolist())[::-1]
    order = order[interval:]  # ignore first few nans
    order = [idx for idx in order if idx < anomaly_range[0]]  # only take the patterns before the anomaly
    print(order)

    result = []

    max_idx = order[0]
    match = df[max_idx - interval + 1: max_idx + 1].copy()
    match['range'] = f'{match["date"].iloc[0]}-{match["date"].iloc[-1]}'
    print(match)

    for var in var_columns:
        var_df = match[[v for v in match.columns if v.endswith(var)] + ['date', 'range']].copy()
        var_df.drop(f'corr_{var}', axis=1, inplace=True)
        var_df.rename(columns={var: 'value', f'label_{var}': 'label', f'score_{var}': 'score'}, inplace=True)
        var_df['name'] = var
        print(var_df)
        data = var_df.to_dict('records')
        print(data)
        result.extend(data)

    print(result)
    return result


def possible_outliers(anomaly_timestamp: str, interval: int) -> List[dict]:
    """
    :param anomaly_timestamp: timestamp of the anomaly to calculate outliers for
    :param interval: length of the data for box/violin plot
    :return: the close pattern found as a list of rows
    """
    firestore = Firestore()
    df = firestore.get_full_data()
    print(df)
    anomaly_idx = df.index[df['date'] == anomaly_timestamp][0]
    anomaly_range = [anomaly_idx - floor((interval - 1) / 2), anomaly_idx + ceil((interval - 1) / 2)]
    if anomaly_range[0] < 0:
        anomaly_range[1] = anomaly_range[1] - anomaly_range[0]
        anomaly_range[0] = max(0, anomaly_range[0])
    elif anomaly_range[1] >= len(df):
        anomaly_range[0] = anomaly_range[0] - (anomaly_range[1] - len(df) + 1)
        anomaly_range[1] = min(anomaly_range[1], len(df) - 1)
    anomaly = df.loc[anomaly_range[0]:anomaly_range[1]]
    print(anomaly)
    var_columns = [c for c in df.columns if (not c.startswith('label') and not c.startswith('score') and c not in ['date'])]

    result = []
    for row in anomaly.to_dict('records'):
        for v in var_columns:
            data = {
                'name': v,
                'value': row[v]
            }
            result.append(data)
    print(result)
    return result


def score_heatmap(length: int):
    """
    :param length: length of data in timestamps to request
    :return: most recent {length} timestamps of data
    """
    firestore = Firestore()
    df = firestore.get_full_data()
    df = df.tail(length)
    var_columns = [c for c in df.columns if (not c.startswith('label') and not c.startswith('score') and c not in ['date'])]

    df['score_avg'] = df[[f'score_{v}' for v in var_columns]].mean(axis=1)
    print(df)

    result = []
    for row in df.to_dict('records'):
        for v in var_columns:
            data = {
                'date': row['date'],
                'score': row[f'score_{v}'],
                'name': v
            }
            result.append(data)
        data = {
            'date': row['date'],
            'score': row['score_avg'],
            'name': 'Overall'
        }
        result.append(data)
    print(result)
    return result


def anomaly_details(variable_name: str, anomaly_timestamp: str, interval: int, count: int):
    close_patterns = close_pattern_chart(variable_name, anomaly_timestamp, interval, count)
    outliers = possible_outliers(anomaly_timestamp, interval)
    data = {
        'anomaly_id': anomaly_timestamp,
        'close_patterns': close_patterns,
        'possible_outliers': outliers
    }
    firestore = Firestore()
    firestore.add_anomaly_details(anomaly_timestamp, data)
