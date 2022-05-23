from math import floor, ceil
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from server_demo.firestore import Firestore


def stats():
    firestore = Firestore()
    df = firestore.get_full_data()
    result = {}
    for var in df.columns:
        if not var.startswith('label') and var not in ['date']:
            result[var] = df[f'label_{var}'].sum()
    return {
        'total': {
            'n': df['label'].sum(),
            'percent': df['label'].sum() / len(df)
        },
        'variable': [
            {'name': k, 'n': v, 'percent': v / sum(result.values()) * 100} for k, v in result.items()
        ]
    }


def hourly_chart():
    firestore = Firestore()
    df = firestore.get_full_data()
    df = df[df['label'] == 1]
    df['date'] = pd.to_datetime(df['date'])
    df['hour'] = df['date'].dt.hour
    result = df['hour'].value_counts()
    result = [{'time': f'{k}:00', 'value': v} for k, v in result.to_dict().items()]
    return result


def weekly_chart():
    firestore = Firestore()
    df = firestore.get_full_data()
    df = df[df['label'] == 1]
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.day_name()
    result = df['day_of_week'].value_counts()
    result = [{'time': k, 'value': v} for k, v in result.to_dict().items()]
    print(result)
    return result


def main_chart(length: int):
    """
    :param length: length of data in timestamps to request
    :return: most recent {length} timestamps of data
    """
    firestore = Firestore()
    df = firestore.get_full_data()
    df = df.tail(length)
    var_columns = [c for c in df.columns if (not c.startswith('label') and not c.startswith('score') and c not in ['date'])]
    data = df.to_dict(orient='list')
    result = []
    for idx in range(len(data['date'])):
        for v in var_columns:
            d = {
                'date': data['date'][idx],
                'value': data[v][idx],
                'name': v,
                'score': data[f'score_{v}'][idx],
                'label': data[f'label_{v}'][idx]
            }
            result.append(d)
    print(result)
    return result


def anomaly_score_chart(length: int):
    """
    :param length: length of data in timestamps to request
    :return: most recent {length} timestamps of data
    """
    firestore = Firestore()
    df = firestore.get_full_data()
    df = df.tail(length)
    df = df[['date', 'score']]
    print(df.to_dict(orient='records'))
    return df.to_dict(orient='records')


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
    pass
