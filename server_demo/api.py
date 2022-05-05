@app.route('/', methods=['GET'])
def close_pattern_of_variable(variable_name: str, anomaly_timestamp: str, interval: int, count: int):
    """
    :param variable_name: name of the variable to search for
    :param anomaly_timestamp: timestamp of the anomaly to find similar patterns for
    :param interval: length of each pattern found
    :param count: the number of similar patterns to find
    :return: an array of similar patterns
    """
    return {

    }

def close_pattern_in_past(anomaly_timestamp:str, interval:int):
    pass