"""Configs for the augment.
"""
from pathlib import Path
import torch

PROJECT_NAME = 'edansa'
DATASET_NAME_V = 'edansa_v5'
#   DEFAULT values,
# might be changed by command arguments or by setting wandb args
default_config = {
    'project_name': PROJECT_NAME,
    'dataset_name_v': DATASET_NAME_V,
    'batch_size': 32,
    'epochs': 500,
    'patience': -1,
    'learning_rate': 1e-3,  # default 1e-3,
    'weight_decay': 1e-2,  # pytorch default 1e-2,
    'device': 0,
    'run_id_2resume': '',  # wandb run id to resume
    'checkpointfile_2resume': '',  # full path
    'checkpoint_every_Nth_epoch': 100,
    'checkpoint_metric': 'val_f1_min',  # 'val_AUC_min', 'val_loss'
    'augmentations': {  # augmentation params
        'spec_augmenter': True,
        'random_mergev2': True,
        'random_merge_fair': False,
        'AddGaussianNoise': True,
        'gauss_max_amplitude': 0.015,
        'mix_channels': False,
        'mix_channels_coeff': 0.3
    },
    'arch': {
        'feature_method': 'logmel',  # 'logmel', 'mfcc', 'spectrogram'
        'intermediate_pool_type': 'max',  # avg+max | avg | max | 
        'global_pool_type': 'avg+max',  # avg+max | avg | max | empty string
        'loss': 'BCEWithLogitsLoss',  # 'CrossEntropyLoss', 'BCEWithLogitsLoss'
        'loss_weights': [],  # [] or [1,1] for no weights,
    },
}

default_config['wandb_mode'] = 'online'  # offline

default_config['load_clipping'] = True

# audio.divide_long_sample ignores less than 5 seconds
#   if file longer than 10 seconds but have extra seconds
#   ex: if it is 54 seconds
#   we get 5 samples and 4 seconds is ignored
default_config['sample_length_limit'] = 2
default_config['dataset_in_memory'] = True
default_config['sampling_rate'] = 48000

default_config['channels'] = 1
default_config['exp_dir'] = ('./')
default_config['excerpt_length'] = 10
default_config['max_mel_len'] = 938  # old 850
default_config['audio_dtype'] = torch.float32

default_config['taxonomy_file_path'] = Path(
    '../../assets/taxonomy/taxonomy_V2.yaml')
default_config['dataset_cache_folder'] = Path(f'./{DATASET_NAME_V}/')

default_config['dataset_folder'] = None  # './assets/EDANSA-2019/data/'
default_config['audio_data_cache_path'] = ''
default_config['dataset_csv_path'] = ('../../assets/labels.csv')

# FILES_AS_NP_FILTERED = DATASET_FOLDER / 'files_as_np_filtered_v1.pkl'

default_config['ignore_files'] = list(set([]))
default_config['target_taxo'] = [
    '1.0.0',  # bio 1.71
    '1.1.0',  # bird 2.42
    '1.1.10',  # songbirds 4.38
    '1.1.7',  # duck-goose-swan 15.315
    '0.0.0',  # anthrophony  3.52
    '1.3.0',  # insect, bug  5.42
    '1.1.8',  # grouse-ptarmigan 24.84
    '0.2.0',  # aircraft 6.36
    '3.0.0',  # sil 8.02
    # '2.0.0',  # geo
    # '2.1.0',  # rain
    # '2.3.0',  # wind
]

default_config['excell_names2code'] = {
    'Sil': '3.0.0',
    'Bio': '1.0.0',
    'Airc': '0.2.0',  # aircraft
    # 'Rain': '2.1.0',
    'Grous': '1.1.8',  # grouse-ptarmigan
    'Bug': '1.3.0',  # insect
    'SongB': '1.1.10',  # songbirds
    'DGS': '1.1.7',  # duck-goose-swan
    'Anth': '0.0.0',  # anthrophony
    # 'Geo': '2.0.0',
    'Bird': '1.1.0',
    # 'Wind': '2.3.0'
}

default_config['category_count'] = len(default_config['target_taxo'])
default_config['code2excell_names'] = {
    v: k for k, v in default_config['excell_names2code'].items()
}

print([
    default_config['code2excell_names'][x]
    for x in default_config['target_taxo']
])

not_original_train_set = [('anwr', '35'), ('anwr', '42'), ('anwr', '43'),
                          ('dalton', '01'), ('dalton', '02'), ('dalton', '03'),
                          ('dalton', '04'), ('dalton', '05'), ('dalton', '06'),
                          ('dalton', '07'), ('dalton', '08'), ('dalton', '09'),
                          ('dalton', '10'), ('dempster', '11'),
                          ('dempster', '12'), ('dempster', '13'),
                          ('dempster', '14'), ('dempster', '15'),
                          ('dempster', '16'), ('dempster', '17'),
                          ('dempster', '19'), ('dempster', '20'),
                          ('dempster', '21'), ('dempster', '22'),
                          ('dempster', '23'), ('dempster', '24'),
                          ('dempster', '25'), ('ivvavik', 'ar01'),
                          ('ivvavik', 'ar02'), ('ivvavik', 'ar03'),
                          ('ivvavik', 'ar04'), ('ivvavik', 'ar05'),
                          ('ivvavik', 'ar06'), ('ivvavik', 'ar07'),
                          ('ivvavik', 'ar08'), ('ivvavik', 'ar09'),
                          ('ivvavik', 'ar10'), ('ivvavik', 'sinp01'),
                          ('ivvavik', 'sinp02'), ('ivvavik', 'sinp03'),
                          ('ivvavik', 'sinp04'), ('ivvavik', 'sinp05'),
                          ('ivvavik', 'sinp06'), ('ivvavik', 'sinp07'),
                          ('ivvavik', 'sinp08'), ('ivvavik', 'sinp09'),
                          ('ivvavik', 'sinp10'), ('prudhoe', '23'),
                          ('prudhoe', '28')]

loc_per_set_train = not_original_train_set + [('anwr', '41'), ('prudhoe', '21'),
                                              ('anwr', '49'), ('anwr', '48'),
                                              ('prudhoe', '19'),
                                              ('prudhoe', '16'), ('anwr', '39'),
                                              ('prudhoe', '30'), ('anwr', '38'),
                                              ('prudhoe', '22'),
                                              ('prudhoe', '11'), ('anwr', '37'),
                                              ('anwr', '44'), ('anwr', '33'),
                                              ('prudhoe', '29'), ('anwr', '46'),
                                              ('prudhoe', '25'),
                                              ('prudhoe', '13'),
                                              ('prudhoe', '24'),
                                              ('prudhoe', '17'), ('anwr', '40'),
                                              ('prudhoe', '14')]
loc_per_set_val = [('prudhoe', '15'), ('prudhoe', '20'), ('anwr', '31'),
                   ('anwr', '47'), ('anwr', '34')]

loc_per_set_test = [('prudhoe', '12'), ('prudhoe', '27'), ('prudhoe', '26'),
                    ('anwr', '45'), ('anwr', '50'), ('prudhoe', '18'),
                    ('anwr', '32'), ('anwr', '36')]

default_config['loc_per_set'] = {
    'train': loc_per_set_train,
    'valid': loc_per_set_val,
    'test': loc_per_set_test
}

if default_config['arch']['loss_weights']:
    assert len(default_config['arch']['loss_weights']) == len(
        default_config['target_taxo'])
