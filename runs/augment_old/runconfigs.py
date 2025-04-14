"""Configs for the augment.
"""
from pathlib import Path

PROJECT_NAME = 'edansa'
DATASET_NAME_V = 'edansa_v5'
#   DEFAULT values,
# might be changed by command arguments or by setting wandb args
default_config = {
    'batch_size': 32,
    'epochs': 1500,
    'patience': -1,
    'device': 0,
    'run_id_2resume': '',  # wandb run id to resume
    'checkpointfile_2resume': '',  # full path
    'checkpoint_every_Nth_epoch': 100,
    # augmentation params
    'spec_augmenter': True,
    'random_mergev2': True,
    'random_merge_fair': False,
    'AddGaussianNoise': True,
    'gauss_max_amplitude': 0.015,
    'mix_channels': False,
    'mix_channels_coeff': 0.3,
    'intermediate_pool_type': 'max',  # avg+max | avg | max | 
    'global_pool_type': 'avg+max',  # avg+max | avg | max | empty string
    'FILTER_DATA_BY_SHAPLEY': False,
    'shapley_csv': '/scratch/arsyed/alaska/journal/shaps.csv',
    'shapley_sample_id_csv': (
        'preds/datasetV3.1.1/' +
        'run-20220215_231152-2sqvtwmy_best_model_62_mean_ROC_AUC=0.9005_wcols.csv'
    ),
    'shapley_value_col_id': 'shaps_ml',
    'shapley_filter_percentage': 0.20,
    'shapley_filter_strategy': 'best2worst',
    'FLIP_Y_BY_SHAPLEY': False,
    'shapley_sample_count': 200,
}

TAXO_COUNT_LIMIT = 0

# audio.divide_long_sample ignores less than SAMPLE_LENGTH_LIMIT seconds
#   if file longer than 10 seconds but have extra seconds
#   ex: if it is 54 seconds
#   we get 5 samples and 4 seconds is ignored
SAMPLE_LENGTH_LIMIT = 2
SAMPLES_FIXED_SIZE = True
DATA_FROM_DISK = True
if not SAMPLES_FIXED_SIZE and DATA_FROM_DISK:
    raise ValueError('Cannot use DATA_FROM_DISK without SAMPLES_FIXED_SIZE')

SAMPLING_RATE = 48000
EXP_DIR = Path('./')

CATEGORY_COUNT = 9
EXCERPT_LENGTH = 10
MAX_MEL_LEN = 938  # old 850

TAXONOMY_FILE_PATH = Path('./assets/taxonomy/taxonomy_V2.yaml')

DATASET_CACHE_FOLDER = Path('./datasetV5_cache/')

AUDIO_DATA_CACHE_PATH = ''

DATASET_CSV_PATH = ('./assets/labels.csv')
DATASET_FOLDER = './assets/EDANSA-2019/data/'

IGNORE_FILES = set([
    # 'S4A10268_20190610_103000_bio_anth.wav',  # has two topology bird/plane
])

EXCELL_NAMES2CODE = {
    'anth': '0.0.0',
    'auto': '0.1.0',
    'car': '0.1.2',
    'truck': '0.1.1',
    'prop': '0.2.1',
    'helo': '0.2.2',
    'jet': '0.2.3',
    'mach': '0.3.0',
    'bio': '1.0.0',
    'bird': '1.1.0',
    'crane': '1.1.11',
    'corv': '1.1.12',
    'hum': '1.1.1',
    'shorb': '1.1.2',
    'rapt': '1.1.4',
    'owl': '1.1.6',
    'woop': '1.1.9',
    'bug': '1.3.0',
    'bugs': '1.3.0',
    'insect': '1.3.0',
    'dgs': '1.1.7',
    'flare': '0.4.0',
    'fox': '1.2.4',
    'geo': '2.0.0',
    'grouse': '1.1.8',
    'grous': '1.1.8',
    'loon': '1.1.3',
    'loons': '1.1.3',
    'mam': '1.2.0',
    'bear': '1.2.2',
    'plane': '0.2.0',
    'ptarm': '1.1.8',
    'rain': '2.1.0',
    'seab': '1.1.5',
    'seabirds': '1.1.5',
    'mous': '1.2.1',
    'deer': '1.2.3',
    'woof': '1.2.4',
    'weas': '1.2.5',
    'meow': '1.2.6',
    'hare': '1.2.7',
    'shrew': '1.2.8',
    'fly': '1.3.2',
    'silence': '3.0.0',
    'sil': '3.0.0',
    'songbird': '1.1.10',
    'songb': '1.1.10',
    'birdsong': '1.1.10',
    'unknown': 'X.X.X',
    'water': '2.2.0',
    'x': 'X.X.X',
    'airc': '0.2.0',
    'aircraft': '0.2.0',
    'mosq': '1.3.1',
    'wind': '2.3.0',
    'windy': '2.3.0',
}