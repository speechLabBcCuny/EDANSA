"""Experiment running. Can only be imported from the experiment folder.

"""

import os
import random

import torch
import torch.nn as nn
from torchvision import transforms

import numpy as np

from pathlib import Path
from collections import Counter
import wandb
from ignite.metrics import Loss

#TODO find a more elegant way to do that
# in case edansa is not installed as a package
try:
    # so we can import from main folder
    from runs.augment import runconfigs
    from runs.augment import modelarchs
    from src.edansa import runutils
    from src.edansa import utils
    from src import edansa
except:
    import runconfigs  # type: ignore
    import modelarchs  # type: ignore
    from edansa import runutils
    from edansa import utils
    import edansa

from ignite.contrib.handlers import wandb_logger


def prepare_dataset(dataset_in_memory=True, load_clipping=True):

    taxo_count_limit = runconfigs.TAXO_COUNT_LIMIT
    sample_length_limit = runconfigs.SAMPLE_LENGTH_LIMIT
    taxonomy_file_path = runconfigs.TAXONOMY_FILE_PATH

    dataset_csv_path = runconfigs.DATASET_CSV_PATH

    ignore_files = runconfigs.IGNORE_FILES

    excerpt_length = runconfigs.EXCERPT_LENGTH
    excell_names2code = runconfigs.EXCELL_NAMES2CODE
    dataset_name_v = runconfigs.DATASET_NAME_V
    dataset_cache_folder = runconfigs.DATASET_CACHE_FOLDER

    audio_dataset, deleted_files = edansa.preparedataset.run(  # type: ignore
        dataset_csv_path,
        taxonomy_file_path,
        ignore_files,
        excerpt_length,
        sample_length_limit,
        taxo_count_limit,
        excell_names2code=excell_names2code,
        dataset_name_v=dataset_name_v,
        dataset_cache_folder=dataset_cache_folder,
        load_clipping=load_clipping,
        dataset_folder=runconfigs.DATASET_FOLDER,
    )
    if dataset_in_memory:
        audio_dataset.load_audio_files(runconfigs.AUDIO_DATA_CACHE_PATH)
    # audio_dataset.pick_channel_by_clipping()

    return audio_dataset, deleted_files


def setup_config(config):

    random_seed: int = 42
    # reproducibility results
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.benchmark = False  # type: ignore
    # torch.set_deterministic(True)

    # wandb.init(config=runconfigs.default_config, project=runconfigs.PROJECT_NAME)
    # config = wandb.config
    config = runconfigs.default_config
    # wandb.config.update(args) # adds all of the arguments as config variables
    # config['batch_size'] = 64
    Path(runconfigs.EXP_DIR).mkdir(exist_ok=True, parents=True)
    os.chdir(runconfigs.EXP_DIR)

    device = torch.device(f"cuda:{config['device']}" if
                          torch.cuda.is_available() else "cpu")  # type: ignore

    config['device'] = device

    return config


def put_samples_into_array(audio_dataset, data_by_reference=False):
    # sound_ins[1].taxo_code
    # classA = 1.1.7 #'duck-goose-swan'
    # classB = 0.2.0 # other-aircraft
    # 3.0.0 : 0.48, 0.26, 0.26, 46 # silence
    # 2.1.0 : 0.22, 0.56, 0.22, 18 # rain
    # 1.3.0 1.3.0 : 0.52, 0.4, 0.087, 161 # insect
    # 1.1.8 : 0.49, 0.19, 0.32, 88 # grouse-ptarmigan

    x_data = []
    y = []
    location_id_info = []

    for sound_ins in audio_dataset.values():
        if not data_by_reference:
            for sample in sound_ins.samples:
                y.append(sound_ins.taxo_codes)
                location_id_info.append(
                    (sound_ins.region, sound_ins.location_id))
                x_data.append(sample)
        else:
            y.append(sound_ins.taxo_codes)
            location_id_info.append((sound_ins.region, sound_ins.location_id))
            x_data.append(sound_ins)

    return x_data, y, location_id_info


def create_multi_label_vector(alphabet, y_data):
    # define input string
    # define universe of possible input values
    # alphabet = ['1.1.10','1.1.7']

    # define a mapping of chars to integers
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    # int_to_char = dict((i, c) for i, c in enumerate(alphabet))

    integer_encoded = []
    for taxo_codes in y_data:
        int_values = [char_to_int.get(taxo, None) for taxo in taxo_codes]
        int_values = [x for x in int_values if x is not None]
        integer_encoded.append(int_values)

    onehot_encoded = list()
    #     print(integer_encoded)
    for values in integer_encoded:
        letter = [0 for _ in range(len(alphabet))]
        for value in values:
            #             print(value)
            letter[value] = 1
        onehot_encoded.append(letter)
    return onehot_encoded


def split_train_test_val(x_data,
                         location_id_info,
                         onehot_encoded,
                         loc_per_set,
                         data_by_reference=False):
    X_train, X_test, X_val, y_train, y_test, y_val = [], [], [], [], [], []
    loc_id_train = []
    loc_id_test = []
    loc_id_valid = []

    for sample, y_val_ins, loc_id in zip(x_data, onehot_encoded,
                                         location_id_info):
        if loc_id in loc_per_set[0]:
            X_train.append(sample)
            y_train.append(y_val_ins)
            loc_id_train.append(loc_id)
        elif loc_id in loc_per_set[1]:
            X_test.append(sample)
            y_test.append(y_val_ins)
            loc_id_test.append(loc_id)
        elif loc_id in loc_per_set[2]:
            X_val.append(sample)
            y_val.append(y_val_ins)
            loc_id_valid.append(loc_id)
        # else:
        #     X_train.append(sample)
        #     y_train.append(y_val_ins)
        #     loc_id_train.append(loc_id)
        # print(
        #     f'WARNING: ' +
        #     f'This sample is NOT from a location ({loc_id}) that is from ' +
        #     f'pre-determined training,test,validation locations')
    if not data_by_reference:
        X_train, X_test, X_val = np.array(X_train), np.array(X_test), np.array(
            X_val)

        X_train, X_test, X_val = torch.from_numpy(X_train).float(
        ), torch.from_numpy(X_test).float(), torch.from_numpy(X_val).float()
    # else:
    #     # we can just use reference to the audio instances
    #     pass

    y_train, y_test, y_val = np.array(y_train), np.array(y_test), np.array(
        y_val)
    y_train, y_test, y_val = torch.from_numpy(y_train).float(
    ), torch.from_numpy(y_test).float(), torch.from_numpy(y_val).float()

    return X_train, X_test, X_val, y_train, y_test, y_val


def prepare_run_inputs(config,
                       X_train,
                       X_test,
                       X_val,
                       y_train,
                       y_test,
                       y_val,
                       data_by_reference=False,
                       non_associative_labels=None):

    # using wav as input
    to_tensor = modelarchs.WaveToTensor(runconfigs.MAX_MEL_LEN,
                                        runconfigs.SAMPLING_RATE,
                                        device=config['device'])

    # Transforms to apply per sample
    transformers = [
        to_tensor,
    ]

    if config['spec_augmenter']:
        spec_augmenter = modelarchs.SpecAugmentation(time_drop_width=64,
                                                     time_stripes_num=2,
                                                     freq_drop_width=8,
                                                     freq_stripes_num=2)

        transformers.append(spec_augmenter)

    transformCompose = transforms.Compose(transformers)

    transformCompose_eval = transforms.Compose([
        # Do not add any augmentation for test and val !!!
        to_tensor,
        # Do not add any augmentation for test and val !!!
    ])

    # transforms to be applied per batch
    batch_transforms = []
    if config['random_mergev2']:
        batch_transforms.append('random_mergev2')
    if config['random_merge_fair']:
        batch_transforms.append('random_merge_fair')
    if config['AddGaussianNoise']:
        batch_transforms.append('AddGaussianNoise')

    sound_datasets = {
        phase: runutils.audioDataset(
            XY[0],
            XY[1],
            transform=transformCompose_eval,
            data_by_reference=data_by_reference,
            non_associative_labels=non_associative_labels,
        ) for phase, XY in zip(['val', 'test'],
                               [[X_val, y_val], [X_test, y_test]])
    }

    sound_datasets['train'] = runutils.AugmentingAudioDataset(
        X_train,
        y_train,
        transform=transformCompose,
        sampling_rate=runconfigs.SAMPLING_RATE,
        batch_transforms=batch_transforms,
        gauss_max_amplitude=config['gauss_max_amplitude'],
        data_by_reference=data_by_reference,
        non_associative_labels=non_associative_labels,
    )

    print('train_size', len(sound_datasets['train']))
    print('test_size', len(sound_datasets['test']))
    print('val_size', len(sound_datasets['val']))

    print('train category sizes:', torch.sum(y_train, 0))
    print('test category sizes:', torch.sum(y_test, 0))
    print('val category sizes:', torch.sum(y_val, 0))

    data_loader_params = {
        'batch_size': config['batch_size'],
        'shuffle': True,
        'num_workers': 0
    }

    dataloaders = {
        x: torch.utils.data.DataLoader(sound_datasets[x], **data_loader_params)
        for x in ['train', 'val', 'test']
    }

    model = modelarchs.Cnn6(
        None,
        None,
        None,
        None,
        None,
        None,
        runconfigs.CATEGORY_COUNT,
        intermediate_pool_type=config['intermediate_pool_type'],
        global_pool_type=config['global_pool_type'],
    )
    # device is defined before

    model.float().to(config['device'])  # Move model before creating optimizer
    optimizer = torch.optim.AdamW(  # type: ignore
        model.parameters(),
        #                                 weight_decay=config['weight_decay'],
    )

    criterion = nn.BCEWithLogitsLoss()
    # statHistory={'valLoss':[],'trainLoss':[],'trainAUC':[],'valAUC':[]}

    metrics = {
        'loss':
            Loss(criterion),  # 'accuracy': Accuracy(),
        #     'ROC_AUC': ROC_AUC(runutils.activated_output_transform),
        'ROC_AUC':
            edansa.metrics.ROC_AUC_perClass(  # type: ignore
                edansa.metrics.activated_output_transform),  # type: ignore
    }

    return model, optimizer, dataloaders, metrics, criterion


def load_shapley(shapley_csv, sample_id_csv, filter_key):
    import pandas as pd
    # shapley_csv = '/scratch/arsyed/shapley/alaska-shapley.csv'
    # filter_key = 'shap_songbird'
    shapley_values = utils.read_csv(shapley_csv)
    sample_ids = utils.read_csv(sample_id_csv)
    sample_id2clip_path = {}
    path2shapely = {}
    for row in sample_ids:
        sample_id2clip_path[row['sample_id']] = row['file_path']
    for row in shapley_values:
        path2shapely[sample_id2clip_path[row['sample_id']]] = float(
            row[filter_key])

    return path2shapely


# Delete bottom 93.0% samples
# database size 6495
# database size after filtering 3337
# ['random_mergev2', 'AddGaussianNoise']

# train_size 1283
# test_size 1091
# val_size 963


# select samples according to shapley
def delete_samples_by_shapley(dataset,
                              shapley_values,
                              percentage,
                              strategy='best2worst'):
    '''
    select samples according to shapley

    Args:
        dataset: audio_dataset
        shapley_values: shapley values dict {'Clip Path': shapley_value}
        percentage: percentage of samples to select
        strategy: 'best2worst' or 'worst2best', which one to select
    '''
    # shapley values should have only training samples
    assert len(shapley_values) != len(dataset)
    assert percentage > 0 and percentage <= 1
    if percentage == 1:
        print('No need to select samples, percentage is 1')
        print('database size', len(dataset))
        return dataset

    if strategy == 'best2worst':
        shapley_values = sorted(shapley_values.items(),
                                key=lambda x: x[1],
                                reverse=True)
    elif strategy == 'worst2best':
        shapley_values = sorted(shapley_values.items(), key=lambda x: x[1])
    else:
        raise ValueError('strategy should be best2worst or worst2best')

    len_shapley = len(shapley_values)
    # delete bottom (1-percentage)% samples
    print('Delete bottom {}% samples'.format((1 - percentage) * 100))
    print('database size', len(dataset))
    for (file_path,
         shapley_value) in shapley_values[len_shapley - 1:int(len_shapley *
                                                              (percentage)):-1]:
        del dataset[file_path]
    print('database size after filtering', len(dataset),
          ' (inlcudes validation and test which are not filtered)')
    return dataset


def run_exp(wandb_logger_ins):

    config = wandb_logger_ins.config

    # dataset split by location
    not_original_train_set = [('anwr', '35'), ('anwr', '42'), ('anwr', '43'),
                              ('dalton', '01'), ('dalton', '02'),
                              ('dalton', '03'), ('dalton', '04'),
                              ('dalton', '05'), ('dalton', '06'),
                              ('dalton', '07'), ('dalton', '08'),
                              ('dalton', '09'), ('dalton', '10'),
                              ('dempster', '11'), ('dempster', '12'),
                              ('dempster', '13'), ('dempster', '14'),
                              ('dempster', '15'), ('dempster', '16'),
                              ('dempster', '17'), ('dempster', '19'),
                              ('dempster', '20'), ('dempster', '21'),
                              ('dempster', '22'), ('dempster', '23'),
                              ('dempster', '24'), ('dempster', '25'),
                              ('ivvavik', 'AR01'), ('ivvavik', 'AR04'),
                              ('ivvavik', 'AR06'), ('ivvavik', 'SINP01'),
                              ('ivvavik', 'SINP03'), ('ivvavik', 'SINP04'),
                              ('ivvavik', 'SINP05'), ('ivvavik', 'SINP06'),
                              ('ivvavik', 'SINP07'), ('ivvavik', 'SINP08'),
                              ('ivvavik', 'SINP09'), ('ivvavik', 'SINP10'),
                              ('prudhoe', '23'), ('prudhoe', '28')]
    loc_per_set = [
        not_original_train_set + [('anwr', '41'), ('prudhoe', '21'),
                                  ('anwr', '49'), ('anwr', '48'),
                                  ('prudhoe', '19'), ('prudhoe', '16'),
                                  ('anwr', '39'), ('prudhoe', '30'),
                                  ('anwr', '38'), ('prudhoe', '22'),
                                  ('prudhoe', '11'), ('anwr', '37'),
                                  ('anwr', '44'), ('anwr', '33'),
                                  ('prudhoe', '29'), ('anwr', '46'),
                                  ('prudhoe', '25'), ('prudhoe', '13'),
                                  ('prudhoe', '24'), ('prudhoe', '17'),
                                  ('anwr', '40'), ('prudhoe', '14')],
        [('prudhoe', '15'), ('prudhoe', '20'), ('anwr', '31'), ('anwr', '47'),
         ('anwr', '34')],
        [('prudhoe', '12'), ('prudhoe', '27'), ('prudhoe', '26'),
         ('anwr', '45'), ('anwr', '50'), ('prudhoe', '18'), ('anwr', '32'),
         ('anwr', '36')]
    ]
    target_taxo = [
        '1.0.0',
        '1.1.0',
        '1.1.10',
        '1.1.7',
        '0.0.0',
        '1.3.0',
        '1.1.8',
        '0.2.0',
        '3.0.0',
    ]
    target_taxo_names = {
        '1.0.0': 'biophony',
        '1.1.0': 'bird',
        '1.1.10': 'songbirds',
        '1.1.7': 'duck-goose-swan',
        '0.0.0': 'anthrophony',
        '1.3.0': 'insect',
        '1.1.8': 'grouse-ptarmigan',
        '0.2.0': 'aircraft',
        '3.0.0': 'silence',
    }
    target_taxo_names = [
        target_taxo_names[taxo_code] for taxo_code in target_taxo
    ]

    # label indexs we cannot mix with other labels
    non_associative_labels = [target_taxo.index(i) for i in ['3.0.0']]
    # geo_labels = [target_taxo.index(i) for i in ['2.0.0', '2.1.0', '2.3.0']]

    # 0.0.0 anthrophony
    # 0.2.0 aircraft
    # 1.0.0 biophony
    # 1.1.0 bird
    # 1.1.10 songbirds
    # 1.1.7 duck-goose-swan
    # 1.3.0 insect
    # 1.1.8 grouse-ptarmigan
    # 3.0.0 silence
    print('Preparing Dataset.')
    if runconfigs.DATA_FROM_DISK:
        audio_dataset, _ = prepare_dataset(dataset_in_memory=False,
                                           load_clipping=True)
    else:
        audio_dataset, _ = prepare_dataset(dataset_in_memory=True,
                                           load_clipping=True)

    # if not config['mix_channels'] and not runconfigs.DATA_FROM_DISK:
    #     audio_dataset.pick_channel_by_clipping()
    print('Generating samples')

    if not runconfigs.SAMPLES_FIXED_SIZE:
        audio_dataset.dataset_generate_samples(runconfigs.EXCERPT_LENGTH)

    ##### shapley stuff#####

    if config['FILTER_DATA_BY_SHAPLEY']:
        audio_dataset = delete_samples_by_shapley(
            audio_dataset,
            shapley_values=load_shapley(config['shapley_csv'],
                                        config['shapley_sample_id_csv'],
                                        config['shapley_value_col_id']),
            percentage=config['shapley_filter_percentage'],
            strategy=config['shapley_filter_strategy'])
    # print('load X_Y_LOC_EMBED')
    # x_y_loc_embed=torch.load( runconfigs.X_Y_LOC_EMBED_PATH)
    # y_data=x_y_loc_embed['y_data']
    # x_data=x_y_loc_embed['embeds_array']
    # # x_data=(x_data.detach().numpy())
    # location_id_info=x_y_loc_embed['location_id_info']
    DATA_BY_REFERENCE = runconfigs.DATA_FROM_DISK
    x_data, y_data, location_id_info = put_samples_into_array(
        audio_dataset, data_by_reference=DATA_BY_REFERENCE)

    multi_label_vector = create_multi_label_vector(target_taxo, y_data)

    # if config['FLIP_Y_BY_SHAPLEY']:
    #     flip_index = target_taxo.index('1.1.10')
    #     multi_label_vector_flipped = []
    #     for i, y in enumerate(multi_label_vector):
    #         y_flipped = y[:]
    #         #flip one to zero, zero to one
    #         if i in sample_ids:
    #             y_flipped[flip_index] = 1 - y_flipped[flip_index]
    #         multi_label_vector_flipped.append(y_flipped)
    #     multi_label_vector = multi_label_vector_flipped

    X_train, X_test, X_val, y_train, y_test, y_val = split_train_test_val(
        x_data,
        location_id_info,
        multi_label_vector,
        loc_per_set,
        data_by_reference=DATA_BY_REFERENCE)

    model, optimizer, dataloaders, metrics, criterion = prepare_run_inputs(
        config,
        X_train,
        X_test,
        X_val,
        y_train,
        y_test,
        y_val,
        data_by_reference=DATA_BY_REFERENCE,
        non_associative_labels=non_associative_labels)

    print('ready ?')
    checkpoints_dir = runconfigs.EXP_DIR / 'checkpoints'
    checkpoints_dir.mkdir(exist_ok=True)
    print('MODEL:----start-----')
    print(model)
    print('MODEL:------end-----')

    # use wandb folder
    # checkpoints_dir = None

    runutils.run(model,
                 dataloaders,
                 optimizer,
                 criterion,
                 metrics,
                 config['device'],
                 config,
                 runconfigs.PROJECT_NAME,
                 checkpoints_dir=checkpoints_dir,
                 wandb_logger_ins=wandb_logger_ins,
                 taxo_names=target_taxo_names)


def main():
    wandb_project_name = runconfigs.PROJECT_NAME
    default_config = runconfigs.default_config
    config = setup_config(default_config)
    config['run_id_2resume'] = config.get('run_id_2resume', '')
    config['checkpointfile_2resume'] = config.get('checkpointfile_2resume', '')

    if config['run_id_2resume'] == '':
        run_id = wandb.util.generate_id()
    else:
        run_id = config['run_id_2resume']
        print(f'run id found to be RESUMED!: {run_id}')

    if config['run_id_2resume'] != '' and config['checkpointfile_2resume'] == '':
        raise Exception(
            'We need both run_id_2resume and checkpointfile_2resume to resume')
    elif config[
            'run_id_2resume'] == '' and config['checkpointfile_2resume'] != '':
        raise Exception(
            'We need both run_id_2resume and checkpointfile_2resume to resume')

    wandb_logger_ins = wandb_logger.WandBLogger(
        project=wandb_project_name,
        # name=runconfigs.PROJECT_NAME,
        id=run_id,
        config=config,
        resume='allow',
    )

    run_exp(wandb_logger_ins)
    # return audio_dataset, config


if __name__ == '__main__':
    # do not upload model files
    main()
