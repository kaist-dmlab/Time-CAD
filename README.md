# Time-CAD
# Time-Series Anomaly Detection with Context-Aware Decomposition

This is the submission version of the paper.

## 1. Overview
Automatically detecting anomalies in massive time series has become increasingly challenging as data collection technology developed, especially in communication services. Time-series decomposition can reveal underlying time series patterns, such as trends and seasonality. However, previous decomposition-based anomaly detectors did not take temporal auxiliary information (e.g., holidays) into account, limiting their ability to respond to contextual cases. For example, a sharp increase in the value of a given variable might be normal on holidays but abnormal on weekdays. This study proposes a framework for detecting anomalies through deep time-series decomposition by exploiting temporal auxiliary information. To verify the effectiveness of the proposed framework, we conduct thorough experiments on both real-world public and proprietary datasets. The results empirically ascertain that detecting anomalies using the residuals from context-based decomposition improves the performance by up to 2.1 times in time-series aware F1 score.

## 2. Public Data Sets
| Name        | # Timestamp  | # Train  | # Test    | Entity×Dimension | # Anomaly      | Link           |
| :--------:  | :----------: | :------: | :-------: |:----------------:| :------------: |:--------------:|
| Samsung     | 34,902       | 21.600   |  13,302   |  3 × 8           | 160(0.46%)     |Private         |
| KPI         | 111,370      | 66,822   |  44,548   |  1 × 1           | 1,102(0.99%)   |[link](https://github.com/NetManAIOps/KPI-Anomaly-Detection) |
| Energy      | 47,003       | 41,654   |  5,349    |  1 × 1           | 2,772(5.90%)   |[link](https://aihub.or.kr/aidata/30759) |
| IoT-Modbus  | 51,106       | 15,332   |  35,774   |  1 × 4           | 16,106(31.51%) |[link](https://research.unsw.edu.au/projects/toniot-datasets) |

# Requirements and Installations
- [Node.js](https://nodejs.org/en/download/): 16.13.2+
- [Anaconda 4](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) or [Miniconda 3](https://docs.conda.io/en/latest/miniconda.html)

> After installing Node.js and Anaconda, run the following commands on your terminal `sh install.sh && run.sh` to set up dependencies and run the web application.
