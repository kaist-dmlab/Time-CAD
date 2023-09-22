# Time-CAD: Context-Aware Deep Time-Series Decomposition for Anomaly Detection in Businesses

This is the implementation of a paper published in ECML PKDD 2023 (ADS Track) [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-43427-3_20)] [[Poster](https://drive.google.com/file/d/1pc9TtAlnm-RbeQlwgsc2eMNIMLUwDsmN/view?usp=drive_link)] [[Presentation](https://drive.google.com/file/d/1pxG7E8gIt9thUaMg37d-IHHf_w9xY2bn/view?usp=drive_link)]


## 0. Citation
```
@inproceedings{TimeCAD_ECMLPKDD_23,
  title={Context-Aware Deep Time-Series Decomposition for Anomaly Detection in Businesses},
  author={Nam, Youngeun and Trirat, Patara and Kim, Taeyoon and Lee, Youngseop and Lee, Jae-Gil},
  booktitle={Proceedings of the 2023 Joint European Conference on Machine Learning and Knowledge Discovery in Databases},
  pages={330--345},
  year={2023}
}
```

## 1. Overview
Detecting anomalies in time series has become increasingly challenging as data collection technology develops, especially in realworld communication services, which require contextual information for precise prediction. To address this challenge, researchers usually use time-series decomposition to reveal underlying patterns, e.g., trends and seasonality. However, existing decomposition-based anomaly detectors do not explicitly consider such *contextual information*, limiting their ability to correctly detect contextual cases. This paper proposes *Time-CAD*, a new *context-aware deep* time-series decomposition framework to detect anomalies for a more practical scenario in real-world businesses. We verify the effectiveness of the novel design for integrating contextual information into deep time-series decomposition through extensive experiments on four real-world benchmarks, demonstrating improvements of up to 46% in time-series aware $F_1$ score on average.

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
