from typing import List
import pandas as pd

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


def close_pattern_of_variable(variable_name: str, anomaly_timestamp: str, interval: int, count: int) -> List[List[dict]]:
    """
    :param variable_name: name of the variable to search for
    :param anomaly_timestamp: timestamp of the anomaly to find similar patterns for
    :param interval: length of each pattern found
    :param count: the number of similar patterns to find
    :return: an array of similar patterns as a list of list of rows
    """
    firestore = Firestore()
    df = firestore.get_full_data()
    result = df.iloc[0:interval, variable_name]

    return [result.to_dict('records')]


def close_pattern_in_past(anomaly_timestamp: str, interval: int) -> List[dict]:
    """

    :param anomaly_timestamp: timestamp of the anomaly to find similar patterns for
    :param interval: length of the pattern found
    :return: the close pattern found as a list of rows
    """
    firestore = Firestore()
    df = firestore.get_full_data()
    result = df.iloc[0:interval]

    return result.to_dict('records')
