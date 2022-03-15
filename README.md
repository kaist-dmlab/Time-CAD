# Time-CAD
# Time-Series Anomaly Detection with Context-Aware Decomposition

This is the submission version of the paper, KDD 2022 (Applied Data Science track, under review).

## 1. Overview
Automatically detecting anomalies in massive time series has become increasingly challenging as data collection technology developed, especially in communication services. Time-series decomposition can reveal underlying time series patterns, such as trends and seasonality. However, previous decomposition-based anomaly detectors did not take temporal auxiliary information (e.g., holidays) into account, limiting their ability to respond to contextual cases. For example, a sharp increase in the value of a given variable might be normal on holidays but abnormal on weekdays. This study proposes a framework for detecting anomalies through deep time-series decomposition by exploiting temporal auxiliary information. To verify the effectiveness of the proposed framework, we conduct thorough experiments on both real-world public and proprietary datasets. The results empirically ascertain that detecting anomalies using the residuals from context-based decomposition improves the performance by up to 2.1 times in time-series aware F1 score.

## 2. Public Data Sets
| Name        | # Timestamp  | # Train  | # Test    | Entity×Dimension | # Anomaly       | Link           |
| :--------:  | :----------: | :------: | :-------: |:----------------:| :------------:  |:--------------:|
| Samsung     | 34,902       | 21.600   |  13,302   |  3 × 8           | 160 (0.46%)     |Private         |
| KPI         | 111,370      | 66,822   |  44,548   |  1 × 1           | 1,102 (0.99%)   |[link](https://github.com/NetManAIOps/KPI-Anomaly-Detection) |
| Energy      | 47,003       | 41,654   |  5,349    |  1 × 1           | 2,772 (5.90%)   |[link](https://aihub.or.kr/aidata/30759) |
| IoT-Modbus  | 51,106       | 15,332   |  35,774   |  1 × 4           | 16,106 (31.51%) |[link](https://research.unsw.edu.au/projects/toniot-datasets) |

## 3. Requirements and Installations
- [Node.js](https://nodejs.org/en/download/): 16.13.2+
- [Anaconda 4](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) or [Miniconda 3](https://docs.conda.io/en/latest/miniconda.html)
- Python 3.8.12 (Recommend Anaconda)
- Ubuntu 16.04.3 LTS
- tensorflow >= 2.5.0


## 4. Configuration
Time-CAD algorithm was implemented in **Python 3.8.12.**
- Edit main.py files to set experiment parameters (model, dataset, gpu_id(e.g. 0,1,2,3,4,5), etc.)
```
python3 main.py
```

## 5. How to run
- Parameter options
```
--model: the name of model (string)
--dataset: the name of dataset (string)
--seq_length: the size of a window (integer)
--stride: the size of a slide (integer)

--lamda_t: the weight of temporal auxiliary information in [0,1] (integer) 
--wavelet_num: the iteration number of wavelet transformation (integer)
--decomposition: whether to use time series decomposition or not. "1" (true) or "0" (false). (integer, default: 0(false))
--segmentation: whether to evaluate with segmentation-based metrics or not. "1" (true) or "0" (false). (integer, default: 0(false))
```

- At current directory which has all source codes, run main.py with parameters as follows.
```
- model: {MS-RNN CNN GRU Bi-GRU LSTM}   # designate which dataset to use.
- dataset: {samsung, energy, kpi, IoT_modbus}
- seed: {0, 1, 2}                       # seed for 3-fold cross validation.
- gpu_id: an integer for gpu id.
- decomposition: {0, 1}                 # whether to use time series decomposition or not.
- segmentation: {0, 1}                  # whether to evaluate with segmentation-based metrics or not.
e.g.) python3 main.py --seed 0 --model Bi-GRU --dataset IoT_modbus --decomposition 1 --segmentation 1 --gpu_id 0
```
