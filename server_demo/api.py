from typing import List

from server_demo.firestore import Firestore


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
