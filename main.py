import argparse
import os
from train import train
from model import *

model_dict = {"CNN": CNN_AE, "GRU": GRU_AE, "Bi-GRU": BiGRU_AE, "LSTM": LSTM_AE, "MS-RNN": Modified_S_RNN}
temporal_model_dict = {"CNN": CNN_AE_Temporal, "GRU": GRU_AE_Temporal, "Bi-GRU": BiGRU_AE_Temporal, "LSTM": LSTM_AE_Temporal, "MS-RNN": Modified_S_RNN}
seq_length_dict = {"samsung": 36, "energy": 60, "kpi": 60, "unsw": 60, "IoT_fridge": 60, "IoT_modbus": 60}
stride_dict = {"samsung": 1, "energy": 1, "kpi": 1, "unsw": 1, "IoT_fridge": 1, "IoT_modbus": 1}

parser = argparse.ArgumentParser(description='Settings for AD-RTX')
# basic settings
parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train (default: 100)')
parser.add_argument('--seed', type=int, default=0, help='seed number for reapet experiments(default: 0)')
parser.add_argument('--gpu_id', default='1', type=str, help='gpu_ids: e.g. 0, 1')

# basic hyper-parameters
parser.add_argument('--model', type=str, default='CNN', help='model name')
parser.add_argument('--dataset', type=str, default='samsung', help='dataset name')
parser.add_argument('--lamda_t', type=int, default=-0.7, help='weight of temporal auxiliary information')
parser.add_argument('--wavelet_num', type=int, default=3, help='iteration number of wavelet transformation')
parser.add_argument('--temporal', type=int, default=0, help='use temporal information')
parser.add_argument('--decomposition', type=int, default=0, help='use time series decomposition')
parser.add_argument('--segmentation', type=int, default=0, help='evaluate with segmentation-based metrics')

opts = parser.parse_args()

SEED = opts.seed
MODEL = opts.model
DATASET = opts.dataset
TEMPORAL = opts.temporal
DECOMPOSITION = opts.decomposition
SEGMENTATION = opts.segmentation

# set gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ['CUDA_VISIBLE_DEVICES']= opts.gpu_id

def main():

    print("########" ,SEED,"th training start", "########")

    # check settings
    print(f"GPU_ID: {opts.gpu_id}\nSEED: {SEED}")
    print(f"TEMPORAL: {TEMPORAL}\nDECOMPOSITION: {DECOMPOSITION}\nSEGMENTATION: {SEGMENTATION}\nDATASET: {DATASET}\nMODEL: {MODEL}")

    train(
        model_dict[MODEL],
        temporal_model_dict[MODEL],
        MODEL,
        seq_length_dict[DATASET],
        stride_dict[DATASET],
        opts.lamda_t,
        opts.wavelet_num,
        SEED,
        DATASET,
        TEMPORAL,
        DECOMPOSITION,
        SEGMENTATION
        )

if __name__ == '__main__':
    main()