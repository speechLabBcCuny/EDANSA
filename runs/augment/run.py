"""Experiment running. Can only be imported from the experiment folder.

"""

from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms

from ignite.metrics import Loss
from ignite.contrib.handlers import wandb_logger

import runconfigs
from edansa import runutils, datasets, modelarchs, runargs, weather
import edansa.preparedataset
from edansa.dataimport import RecordingsDataset, RecordingsDatasetSSL
import edansa.metrics
import edansa
import inference

from edansa import io as eio


def prepare_dataset(
    config,
    dataset_in_memory=True,
    load_clipping=True,
):

    audio_dataset, deleted_files = edansa.preparedataset.run(  # type: ignore
        config['dataset_csv_path'],
        config['taxonomy_file_path'],
        config['ignore_files'],
        config['excerpt_length'],
        config['sample_length_limit'],
        excell_names2code=config['excell_names2code'],
        dataset_name_v=config['dataset_name_v'],
        dataset_cache_folder=config['dataset_cache_folder'],
        dataset_folder=config['dataset_folder'],
        load_clipping=load_clipping,
        target_taxo=config['target_taxo'])
    training_with_rawrecordings = config.get('RecordingsDataset',
                                             {}).get('active', False)
    if training_with_rawrecordings:
        audio_dataset = runutils.remove_train_data_from_dataset(
            audio_dataset, config)
    if config['debug']:
        audio_dataset = runutils.shorten_dataset4debug(
            audio_dataset,
            config=config,
            counts=[128, 64, 256],
            ignore_train=training_with_rawrecordings)
    if dataset_in_memory:

        audio_dataset.load_audio_files(
            config['audio_data_cache_path'],
            resample_rate=config['sampling_rate'],
            dtype=torch.float32,
            channels=1,
            use_threads=True,
        )
    # audio_dataset.pick_channel_by_clipping()

    return audio_dataset, deleted_files


def get_transforms(config):

    # using wav as input
    to_tensor = modelarchs.WaveToTensor(
        config['max_mel_len'],
        config['sampling_rate'],
        device=config['device'],
        feature_method=config['arch']['feature_method'])

    # Transforms to apply per sample
    transformers = [
        to_tensor,
    ]

    if config['augmentations']['spec_augmenter']:
        spec_augmenter = modelarchs.SpecAugmentation(time_drop_width=64,
                                                     time_stripes_num=2,
                                                     freq_drop_width=8,
                                                     freq_stripes_num=2)

        transformers.append(spec_augmenter)  # type: ignore

    transform_compose = transforms.Compose(transformers)

    transform_compose_eval = transforms.Compose([
        # Do not add any augmentation for test and val !!!
        to_tensor,
        # Do not add any augmentation for test and val !!!
    ])

    # transforms to be applied per batch
    batch_transforms = []
    augmentation_config = config['augmentations']
    if augmentation_config['random_mergev2']:
        batch_transforms.append('random_mergev2')
    # if augmentation_config['random_merge_fair']:
    #     batch_transforms.append('random_merge_fair') # Keep commented or remove
    if augmentation_config['AddGaussianNoise']:
        batch_transforms.append('AddGaussianNoise')

    return transform_compose, transform_compose_eval, batch_transforms


def get_datasets(config,
                 X_train,
                 X_test,
                 X_val,
                 y_train,
                 y_test,
                 y_val,
                 data_by_reference=False,
                 non_associative_labels=None,
                 audio_dtype=torch.float32):

    transform_compose, transform_compose_eval, batch_transforms = get_transforms(
        config)

    sound_datasets = {}
    for phase, XY in zip(['val', 'test'], [[X_val, y_val], [X_test, y_test]]):
        if len(XY[1]) == 0:
            print(f'No {phase} data')
        else:
            print(f'{phase} data count: {len(XY[0])}')
            sound_datasets[phase] = datasets.audioDataset(
                XY[0],
                XY[1],
                transform=transform_compose_eval,
                data_by_reference=data_by_reference,
                non_associative_labels=non_associative_labels,
                xdtype=audio_dtype,
                device=config['device'],
                # channels=config['channels'],
            )

    # if training_with_rawrecordings is true
    # we remove the training data from the dataset in the prepare_dataset function
    # so we need to load the X_train/recordings again
    if len(X_train) == 0:
        exp_type = config.get('RecordingsDataset',
                              {}).get('exp_type', 'weather')
        metadata_path = config.get('RecordingsDataset',
                                   {}).get('metadata_path', None)

        recordings = eio.AudioRecordingFiles(dataframe=metadata_path,
                                             version='v2')
        regloc_list = config['loc_per_set']['train']
        in_memory_sample_count = config.get('RecordingsDataset',
                                            {}).get('in_memory_sample_count',
                                                    10000)

        if config['debug']:
            regloc_list = regloc_list[:2]
            in_memory_sample_count = 100
        if exp_type == 'weather':
            merra_weather = weather.load_all_merra_data()
            X_train = RecordingsDataset(
                recordings=recordings,
                in_memory_sample_count=in_memory_sample_count,
                weather_data=merra_weather,
                regloc_list=regloc_list)
        elif exp_type == 'ssl':
            X_train = RecordingsDatasetSSL(
                recordings=recordings,
                in_memory_sample_count=in_memory_sample_count,
                regloc_list=regloc_list)
        else:
            raise ValueError(
                f'Unknown exp_type: ssl or weather, not {exp_type}')
        print(len(X_train))
        y_train = torch.empty(len(X_train),)

    sound_datasets['train'] = datasets.AugmentingAudioDataset(  # type: ignore
        X_train,
        y_train,
        transform=transform_compose,
        sampling_rate=config['sampling_rate'],
        batch_transforms=batch_transforms,
        gauss_max_amplitude=config['augmentations']['gauss_max_amplitude'],
        data_by_reference=data_by_reference,
        non_associative_labels=non_associative_labels,
        xdtype=audio_dtype,
        device=config['device'],
        mono=config['mono'],
    )
    runutils.print_dataset_sizes(sound_datasets, y_train, y_test, y_val)

    return sound_datasets


def get_run_parts(config, sound_datasets):
    data_loader_params = {
        'batch_size': config['batch_size'],
        'shuffle': True,
        'num_workers': 0
    }

    dataloaders = {
        x: torch.utils.data.DataLoader(sound_datasets[x], **data_loader_params)
        for x in sound_datasets.keys()
    }
    fine_tune = config.get('fine_tune', None)
    if fine_tune is not None and fine_tune['run_id']:

        class ArgsPretrain:
            run_identity = config['fine_tune']['run_identity']
            model_id = config['fine_tune']['run_id'] + '-V1'
            model_path = config['fine_tune']['model_path']
            freeze_backbone = config['fine_tune']['freeze_backbone']
            gpu = config['device'].replace('cuda:', '').replace('cpu', '')
            clipping_path = ''
            file_database = ''
            files_metadata = ''
            output_folder = ''
            config_file = None  # loads from wandb if None

        args_pretrain = ArgsPretrain()
        model, _, _, _ = inference.setup(args_pretrain)
        if args_pretrain.freeze_backbone:
            model.requires_grad_(False)
        in_features = model.fc_audioset.in_features
        model.fc_audioset = torch.nn.Linear(in_features,
                                            config['category_count'],
                                            bias=True)
        modelarchs.init_layer(model.fc_audioset)
        model.fc_audioset.requires_grad_(True)
    else:
        model = modelarchs.Cnn6(
            None,
            None,
            None,
            None,
            None,
            None,
            config['category_count'],
            intermediate_pool_type=config['arch']['intermediate_pool_type'],
            global_pool_type=config['arch']['global_pool_type'],
            last_conv_block_out_chan=config['arch'].get(
                'last_conv_block_out_chan', 512))

    model.float().to(config['device'])  # Move model before creating optimizer
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config['learning_rate'],
                                  weight_decay=config['weight_decay'])

    # Binary classification: Use Binary Cross-Entropy (or Categorical Cross-Entropy with 2 classes)
    # Multi-class classification: Use Categorical Cross-Entropy
    # Multi-label classification: Use Binary Cross-Entropy
    if config['arch'].get('loss_weights') is not None and config['arch'].get(
            'loss_weights'):
        loss_weights = torch.tensor(config['arch']['loss_weights']).to(
            config['device'])
    else:
        loss_weights = None
    if config['arch']['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss(pos_weight=loss_weights)
    elif config['arch']['loss'] == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss(weight=loss_weights)
    else:
        raise ValueError('Unknown loss function.')

    metrics = {
        'loss':
            Loss(criterion),
        'ROC_AUC':
            edansa.metrics.ROC_AUC_perClass(  # type: ignore
                edansa.metrics.activated_output_transform),  # type: ignore
        'f1':
            edansa.metrics.F1_Score_with_Optimal_Threshold(
                edansa.metrics.activated_output_transform)
    }

    return model, optimizer, dataloaders, metrics, criterion


def run_exp(wandb_logger_ins):

    config = wandb_logger_ins.config

    print('Preparing Dataset.')
    dataset_in_memory = config['dataset_in_memory']
    audio_dataset, _ = prepare_dataset(
        config,
        dataset_in_memory=dataset_in_memory,
        load_clipping=config['load_clipping'],
    )

    taxo2name = {v: k for k, v in audio_dataset.excell_names2code.items()}
    target_taxo_names = [
        taxo2name[taxo_code] for taxo_code in audio_dataset.target_taxo
    ]

    # label indexs we cannot mix with other labels
    # non_associative_label = '3.0.0'  # ex: silence
    non_associative_label_name = 'Sil'
    if non_associative_label_name in target_taxo_names:
        print(f'Non-associative label: {target_taxo_names}')
        non_associative_labels = [
            target_taxo_names.index(non_associative_label_name)
        ]
    else:
        non_associative_labels = None
    # if not config['augmentations']['mix_channels'] and config['dataset_in_memory']:
    #     audio_dataset.pick_channel_by_clipping()
    print('Generating samples')

    DATA_BY_REFERENCE = not config['dataset_in_memory']
    x_data, y_data, location_id_info = runutils.put_samples_into_array(
        audio_dataset,
        data_by_reference=DATA_BY_REFERENCE,
        target_taxo=audio_dataset.target_taxo,
        y_type='binary')
    multi_label_vector = runutils.create_multi_label_vector(
        audio_dataset.target_taxo, y_data)

    X_train, X_test, X_val, y_train, y_test, y_val = runutils.split_train_test_val(
        x_data,
        location_id_info,
        multi_label_vector,
        config['loc_per_set'],
        data_by_reference=DATA_BY_REFERENCE)

    sound_datasets = get_datasets(config,
                                  X_train,
                                  X_test,
                                  X_val,
                                  y_train,
                                  y_test,
                                  y_val,
                                  data_by_reference=DATA_BY_REFERENCE,
                                  non_associative_labels=non_associative_labels,
                                  audio_dtype=config['audio_dtype'])

    model, optimizer, dataloaders, metrics, criterion = get_run_parts(
        config, sound_datasets)

    print('ready ?')
    checkpoints_dir = Path(config['exp_dir']) / 'checkpoints'
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
                 config['project_name'],
                 checkpoints_dir=checkpoints_dir,
                 wandb_logger_ins=wandb_logger_ins,
                 taxo_names=target_taxo_names)


def main():
    parser = runargs.get_parser()
    args = vars(parser.parse_args())
    config = runconfigs.default_config

    config.update({k: v for k, v in args.items() if v is not None})
    config['debug'] = args.get('debug', False)
    if config['debug']:
        config['wandb_mode'] = 'disabled'
    config = runutils.setup(config)
    config['loc_per_set'] = runutils.make_locs_caseinsensitive(
        config['loc_per_set'])
    wandb_logger_ins = wandb_logger.WandBLogger(
        project=config['project_name'],
        name=args.get('run_name', None),
        id=config['run_id'],
        config=config,
        resume='allow',
        mode=config['wandb_mode'],
    )
    wandb_logger_ins.save(config['dataset_csv_path'], policy='now')

    run_exp(wandb_logger_ins)


if __name__ == '__main__':
    # do not upload model files
    torch.multiprocessing.set_start_method('spawn')
    main()
