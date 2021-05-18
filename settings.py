import logging
import numpy as np
from torch import Tensor


# Data settings
DATA_ROOT = '/home/gmh/dataset/voc_aug/mydata/'
MEAN = Tensor(np.array([0.485, 0.456, 0.406]))
STD = Tensor(np.array([0.229, 0.224, 0.225]))
SCALES = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0)

CROP_SIZE = 513
IGNORE_LABEL = 255

# Model definition
N_CLASSES = 21
N_LAYERS = 101
STRIDE = 8
BN_MOM = 0.1 # default 3e-4
EM_MOM = 0.9
STAGE_NUM = 3

# Training settings
BATCH_SIZE = 16
ITER_MAX = 45000
ITER_SAVE = 3000

LR_DECAY = 10
LR = 9e-3
LR_MOM = 0.9
POLY_POWER = 0.9
WEIGHT_DECAY = 1e-4

DEVICE = 0
DEVICES = [0,1,2,3]

LOG_DIR = './logdir' 
MODEL_DIR = './models'
NUM_WORKERS = 8

logger = logging.getLogger('train')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


TEST_SCALES = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75)

EXP_NAME = 'test'
TEST_SAVE_DIR = './test_results/test_0508/'
TEST_DATA_ROOT = '/home/gmh/dataset/voc_aug/VOCdevkit/VOC2012/JPEGImages/'
