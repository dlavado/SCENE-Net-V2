import os
import sys
from pathlib import Path
import torch


def get_project_root() -> Path:
    return Path(__file__).parent.parent

def get_experiment_path(model, dataset) -> Path:
    return os.path.join(get_project_root(), 'experiments', f"{model}_{dataset}")

def get_experiment_config_path(model, dataset) -> Path:
    return os.path.join(get_experiment_path(model, dataset), 'defaults_config.yml')


ROOT_PROJECT = get_project_root()
TOSH_PATH = "/media/didi/TOSHIBA EXT/"
SSD_PATH = "/media/didi/PortableSSD/"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if "didi" in str(ROOT_PROJECT):
    if os.path.exists(SSD_PATH):
        EXT_PATH = SSD_PATH
    elif os.path.exists(TOSH_PATH):
        EXT_PATH = TOSH_PATH
    elif os.path.exists("/home/didi/Downloads/data/"):
        EXT_PATH = "/home/didi/Downloads/data/"
    else:
        EXT_PATH = "/home/didi/DATASETS/" # google cluster data dir
        # TOSH_PATH = "/home/didi/DATASETS/"
elif 'vulpix' in str(ROOT_PROJECT):
    EXT_PATH = "/data/d.lavado/"
else:
    EXT_PATH = "/home/d.lavado/" #cluster data dir

TS40K_PATH = os.path.join(EXT_PATH, 'TS40K-Dataset/')
TS40K_FULL_PATH = os.path.join(EXT_PATH, 'TS40K-Dataset/TS40K-FULL/')
TS40K_FULL_PREPROCESSED_PATH = os.path.join(EXT_PATH, 'TS40K-Dataset/TS40K-FULL-Preprocessed/')
TS40K_FULL_PREPROCESSED_VOXELIZED_PATH = os.path.join(EXT_PATH, 'TS40K-Dataset/TS40K-FULL-Preprocessed-Voxelized/')
TS40K_FULL_PREPROCESSED_IDIS_PATH = os.path.join(EXT_PATH, 'TS40K-Dataset/TS40K-FULL-Preprocessed-IDIS/')
TS40K_FULL_PREPROCESSED_SMOTE_PATH = os.path.join(EXT_PATH, 'TS40K-Dataset/TS40K-FULL-Preprocessed-SMOTE/')
# TS40K_PATH = os.path.join(EXT_PATH, "TS40K-NEW/TS40K-Sample/")


LAS_RGB_PROCESSED = os.path.join(SSD_PATH, 'TS40K-Dataset/Labelec_LAS_RGB_2024/Processados/')
LAS_RGB_ORIGINALS = os.path.join(SSD_PATH, 'TS40K-Dataset/Labelec_LAS_RGB_2024/Originais/')

EXPERIMENTS_PATH = os.path.join(ROOT_PROJECT, 'experiments')
WEIGHT_SCHEME_PATH = os.path.join(ROOT_PROJECT, 'core/criterions/hist_estimation.pickle')