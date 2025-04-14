'''Utitilieifor running this experiment
'''

import os
from pathlib import Path
import random

import numpy as np
import pandas as pd

from ignite.contrib.handlers import wandb_logger
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator

# from ignite.metrics import Accuracy, Loss
# from ignite.contrib.metrics import ROC_AUC
from ignite.handlers import (ModelCheckpoint, EarlyStopping,
                             global_step_from_engine, Checkpoint, DiskSaver,
                             EpochOutputStore)

from ignite.utils import setup_logger

import torch
import wandb


def activated_output_transform(output):
    y_pred, y = output
    y_pred = torch.exp(y_pred)
    return y_pred, y


def create_evaluators(model,
                      metrics,
                      device,
                      eval_names=('train', 'val', 'test')):
    evaluators = {}
    for eval_name in eval_names:
        evaluator = create_supervised_evaluator(model,
                                                metrics=metrics,
                                                device=device)
        eos = EpochOutputStore()
        eos.attach(evaluator, 'output')
        evaluator.logger = setup_logger(f'{eval_name} Evaluator')
        evaluators[eval_name] = evaluator
    return evaluators


def get_metric_stats(evaluators, metric_name):
    metric_stats = {}
    for eval_name, evaluator in evaluators.items():
        metric_stats[eval_name] = {}
        metric_val = evaluator.state.metrics[metric_name]
        metric_val = np.asarray(metric_val)
        if np.ndim(metric_val) == 0:  # If metric_val is a scalar
            metric_val = np.array([metric_val])  # Convert it to a 1D array
        metric_stats[eval_name]['raw_data'] = metric_val
        metric_stats[eval_name]['mean'] = np.mean(metric_val).item()
        metric_stats[eval_name]['min'] = np.min(metric_val).item()
        metric_stats[eval_name]['max'] = np.max(metric_val).item()
    return metric_stats


def print_metrics(evaluators):
    for eval_name, evaluator in evaluators.items():
        for metric_name, metric in evaluator.state.metrics.items():
            print(f'{eval_name} - {metric_name}: {metric}')


def log_metric_stats(evaluators, wandb_logger_ins, taxo_names, step,
                     current_epoch, best_values, metric_name):
    metric_stats = get_metric_stats(evaluators, metric_name)
    best_val_metric = best_values['val'][metric_name]
    # Update the best metric value and the corresponding epoch if the current metric value is better
    for metric_aggreate in ['mean', 'min']:
        if metric_stats['val'][metric_aggreate] > best_val_metric[
                metric_aggreate]['value']:
            best_val_metric[metric_aggreate]['value'] = metric_stats['val'][
                metric_aggreate]
            best_val_metric[metric_aggreate]['epoch'] = current_epoch
            wandb_logger_ins.log(
                {
                    f'best_{metric_aggreate}_{metric_name}':
                        best_val_metric[metric_aggreate]['value'],
                    f'best_{metric_aggreate}_Epoch':
                        best_val_metric[metric_aggreate]['epoch']
                },
                step=step)

    # Log the metric value for training and validation sets
    for metric_aggreate in ['mean']:  # ['mean', 'min', 'max']
        for eval_name in ['val', 'train']:
            if eval_name in metric_stats:
                wandb_logger_ins.log(
                    {
                        f'{eval_name}_{metric_aggreate}_{metric_name}':
                            metric_stats[eval_name][metric_aggreate]
                    },
                    step=step)

    # Log the metric value for each class in the taxonomy
    if taxo_names is None:
        taxo_names = [f'class_{k}' for k in range(len(metric_stats['val']))]
    for i, taxo_name in enumerate(taxo_names):  # type: ignore
        for eval_name in ['val', 'test']:
            if eval_name in metric_stats:
                try:
                    raw_metric_val = metric_stats[eval_name]['raw_data'][i]
                except IndexError:
                    raw_metric_val = 0
                wandb_logger_ins.log(
                    {
                        f'{eval_name}_{metric_name.lower()}_{taxo_name}':
                            raw_metric_val
                    },
                    step=step)


def score_function_mean_AUC(engine):
    return np.mean(engine.state.metrics['ROC_AUC']).item()


def score_function_loss(engine):
    print('loss', engine.state.metrics['loss'])
    return engine.state.metrics['loss']


def reverse_loss_function(engine):
    return -1 * engine.state.metrics['loss']


def score_function_min_AUC(engine):
    return np.min(engine.state.metrics['ROC_AUC']).item()


def score_function_mean_F1(engine):
    return np.mean(engine.state.metrics['f1']).item()


def score_function_min_F1(engine):
    return np.min(engine.state.metrics['f1']).item()


def get_checkpoint_dir(checkpoints_dir, run_dir, wandb_run_id):
    if checkpoints_dir is None:
        checkpoint_dir_inrun = (Path(run_dir) / 'checkpoints')
        checkpoint_dir_inrun.mkdir(exist_ok=True)
        checkpoint_dir = checkpoint_dir_inrun
    else:
        # Check if wandb_run_id is same as the in the folder name
        wandb_logger_ins_run_dir = Path(run_dir)
        #run-20210429_035224-12tsplqm/files
        run_info_from_dirname = wandb_logger_ins_run_dir.parent.stem.split('-')
        run_timestamp = run_info_from_dirname[1]
        folder_run_id = run_info_from_dirname[-1]

        if str(wandb_run_id) != folder_run_id:
            raise Exception(
                f' ID from wandb_logger_ins.run.dir is not as same as id from API'
                + f'{wandb_run_id},{folder_run_id}, {run_dir}')
        # Create checkpoint dir
        checkpoint_dir = checkpoints_dir / '-'.join(
            (('run', run_timestamp, wandb_run_id)))

    return checkpoint_dir


def add_checkpoint_handlers(
    objects_to_checkpoint,
    evaluators,
    checkpoints_dir,
    wandb_logger_ins,
    checkpoint_every_Nth_epoch,
    checkpoint_score_name,
):
    """Create checkpoint handler for trainer and model.

    Args:
        trainer (ignite.engine.engine.Engine): trainer engine
        model (torch.nn.Module): model to be saved
        optimizer (torch.optim.Optimizer): optimizer to be saved
        checkpoints_dir (str): directory to save checkpoints
        wandb_logger_ins (wandb.wandb_run.Run): wandb logger instance
        checkpoint_every_Nth_epoch (int): how frequently save the model

    """
    if wandb_logger_ins.run.settings.mode == 'disabled':
        checkpoint_dir = checkpoints_dir
    else:
        checkpoint_dir = get_checkpoint_dir(checkpoints_dir,
                                            wandb_logger_ins.run.dir,
                                            wandb_logger_ins.run.id)

    training_checkpoint = Checkpoint(
        to_save=objects_to_checkpoint,
        save_handler=DiskSaver(
            checkpoint_dir,  # type: ignore
            require_empty=False),
        n_saved=2,  # only keep last 2
        global_step_transform=lambda *_: objects_to_checkpoint['trainer'].state.
        epoch,
    )

    objects_to_checkpoint['trainer'].add_event_handler(
        Events.EPOCH_COMPLETED(
            every=checkpoint_every_Nth_epoch),  # how frequently save the model
        training_checkpoint)

    if checkpoint_score_name == 'val_AUC_mean':
        checkpoint_score_function = score_function_mean_AUC
    elif checkpoint_score_name == 'val_AUC_min':
        checkpoint_score_function = score_function_min_AUC
    elif checkpoint_score_name == 'val_loss':
        checkpoint_score_function = reverse_loss_function
    elif checkpoint_score_name == 'val_f1_mean':
        checkpoint_score_function = score_function_mean_F1
    elif checkpoint_score_name == 'val_f1_min':
        checkpoint_score_function = score_function_min_F1
    else:
        raise ValueError(
            f"checkpoint_metric {checkpoint_score_name} not supported")

    model_checkpoint = ModelCheckpoint(
        checkpoint_dir,  # type: ignore
        n_saved=2,
        filename_prefix='best',
        score_function=checkpoint_score_function,
        score_name=checkpoint_score_name,
        create_dir=True,
        # to take the epoch of the `trainer`L
        global_step_transform=global_step_from_engine(
            objects_to_checkpoint['trainer']),
    )
    evaluators['val'].add_event_handler(
        Events.COMPLETED, model_checkpoint,
        {'model': objects_to_checkpoint['model']})

    return model_checkpoint


def resume_from_checkpoint(objects_to_checkpoint, checkpointfile_2resume):

    #https://github.com/pytorch/ignite/blob/0bb3c6c0ac718258aeb0912744f9b8f3d32b7223/examples/mnist/mnist_save_resume_engine.py
    # restore the best model
    print(f"Resume from the checkpoint: {checkpointfile_2resume} !!!! ")

    checkpoint = torch.load(checkpointfile_2resume)
    Checkpoint.load_objects(to_load=objects_to_checkpoint,
                            checkpoint=checkpoint)


def save_best_predictions_and_targets(trainer, taxo_names, evaluators,
                                      checkpoints_dir):
    epoch = trainer.state.epoch
    # print(f"Saving predictions and targets for best model at epoch {epoch}")
    for eval_name, evaluator in evaluators.items():
        if evaluator.state.output is not None:
            # print(f"Saving predictions and targets for {eval_name} evaluator")
            y_preds = []
            y_targets = []
            # y_pred and y are tensors of shape (batch_size, num_classes) in device
            # in device
            for y_pred, y in evaluator.state.output:
                y_preds.append(y_pred)
                y_targets.append(y)

            save_predictions_and_targets(eval_name, taxo_names, epoch,
                                         checkpoints_dir, y_preds, y_targets)
        else:
            print(
                f"Warning: {eval_name} evaluator state output is None. Skipping saving predictions and targets for this evaluator."
            )


def save_predictions_and_targets(eval_name, taxo_names, epoch, checkpoints_dir,
                                 y_preds, y_targets):
    if taxo_names is None:
        taxo_names = [f'label_{k}' for k in range(y_preds.shape[1])]

    pred_columns = [f'pred_{taxo_name}' for taxo_name in taxo_names]
    target_columns = [f'target_{taxo_name}' for taxo_name in taxo_names]

    y_preds = torch.cat(y_preds, dim=0).cpu().numpy()
    y_targets = torch.cat(y_targets, dim=0).cpu().numpy()

    data = np.column_stack((y_targets, y_preds))
    df = pd.DataFrame(data, columns=target_columns + pred_columns)

    df['set'] = eval_name

    save_path = Path(checkpoints_dir) / f"all_predictions_epoch_{epoch}.csv"
    if not os.path.exists(save_path):
        df.to_csv(save_path, index=False, mode='w')
    else:
        df.to_csv(save_path, index=False, mode='a', header=False)


def compute_metrics(engine,
                    wandb_logger_ins,
                    best_values,
                    dataloaders,
                    evaluators,
                    metrics,
                    checkpoints_dir,
                    taxo_names=None):
    for name, evaluator in evaluators.items():
        evaluator.run(dataloaders[name])

    # save predictions and targets if valudation loss is the best
    best_val_loss = best_values['val']['loss']['raw']['value']
    if best_val_loss == -1:
        pass
    else:
        if best_val_loss > evaluators['val'].state.metrics['loss']:
            best_val_loss = evaluators['val'].state.metrics['loss']
            save_best_predictions_and_targets(engine, taxo_names, evaluators,
                                              checkpoints_dir)

    print_metrics(evaluators)
    for metric in metrics:
        # (evaluators, trainer_state_iteration, current_epoch,
        #   best_epoch, taxo_names, best_ROC_AUC)
        if metric == 'loss':
            continue
        log_metric_stats(
            evaluators,
            wandb_logger_ins,
            taxo_names,
            engine.state.iteration,
            engine.state.epoch,
            best_values,
            metric,
        )

    # log epochs seperatly to use in X axis
    wandb_logger_ins.log({'epoch': engine.state.epoch},
                         step=engine.state.iteration)


def run(model,
        dataloaders,
        optimizer,
        criterion,
        metrics,
        device,
        config,
        wandb_project_name,
        run_name=None,
        checkpoints_dir=None,
        wandb_logger_ins=None,
        taxo_names=None):

    del run_name

    trainer = create_supervised_trainer(model,
                                        optimizer,
                                        criterion,
                                        device=device)
    trainer.logger = setup_logger('Trainer')
    training_with_rawrecordings = config.get('RecordingsDataset',
                                             {}).get('active', False)
    eval_names = ['val']
    if not training_with_rawrecordings:
        eval_names.append('train')
    if 'test' in dataloaders:
        eval_names.append('test')
    eval_names = tuple(eval_names)

    evaluators = create_evaluators(model,
                                   metrics,
                                   device,
                                   eval_names=eval_names)
    # best_ROC_AUC -> [mean,min]
    best_values = {
        'val': {
            'ROC_AUC': {
                'mean': {
                    'value': 0,
                    'epoch': 0,
                },
                'min': {
                    'value': 0,
                    'epoch': 0,
                }
            },
            'f1': {
                'mean': {
                    'value': 0,
                    'epoch': 0,
                },
                'min': {
                    'value': 0,
                    'epoch': 0,
                }
            },
            'loss': {
                'raw': {
                    'value': 1e10,
                    'epoch': 0,
                },
            },
        }
    }

    # best_ROC_AUC = [0, 0]
    # best_epoch = [0, 0]
    # best_val_loss = 1e10

    if wandb_logger_ins is None:
        wandb_logger_ins = wandb_logger.WandBLogger(
            project=wandb_project_name,
            # name=run_name,
            config=config,
        )

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        compute_metrics,
        evaluators=evaluators,
        dataloaders=dataloaders,
        wandb_logger_ins=wandb_logger_ins,
        best_values=best_values,
        taxo_names=taxo_names,
        checkpoints_dir=wandb_logger_ins.run.dir,
        metrics=metrics,
    )

    wandb_logger_ins.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED,  # could add (every=100),
        tag='training',  # type: ignore
        output_transform=lambda loss: {'batchloss': loss}  # type: ignore
    )
    if training_with_rawrecordings:
        tag_evaluator = [('validation', evaluators['val'])]
    else:
        tag_evaluator = [('training', evaluators['train']),
                         ('validation', evaluators['val'])]
    for tag, evaluator in tag_evaluator:
        # Attach the logger to the evaluator on the training dataset and log NLL, Accuracy metrics after each epoch
        # We setup `global_step_transform=lambda *_: trainer.state.iteration` to take iteration value
        # of the `trainer`:
        wandb_logger_ins.attach_output_handler(
            evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag=tag,
            # metric_names=['loss', 'ROC_AUC'],  # type: ignore
            metric_names=['loss'],  # type: ignore
            global_step_transform=lambda *_: trainer.state.
            iteration,  # type: ignore
        )

    wandb_logger_ins.attach_opt_params_handler(
        trainer, event_name=Events.EPOCH_COMPLETED, optimizer=optimizer)
    wandb_logger_ins.watch(model, log='all')

    # Setup object to checkpoint
    objects_to_checkpoint = {
        "trainer": trainer,
        "model": model,
        "optimizer": optimizer,
        # "lr_scheduler": lr_scheduler # we do not have scheduler
    }
    add_checkpoint_handlers(
        objects_to_checkpoint,
        evaluators,
        checkpoints_dir,
        wandb_logger_ins,
        config['checkpoint_every_Nth_epoch'],
        checkpoint_score_name=config['checkpoint_metric'],
    )

    if config['patience'] > 1:
        es_handler = EarlyStopping(patience=config['patience'],
                                   score_function=score_function_loss,
                                   trainer=trainer)
        evaluators['val'].add_event_handler(Events.COMPLETED, es_handler)

    if wandb_logger_ins.run.resumed:
        resume_from_checkpoint(objects_to_checkpoint,
                               config['checkpointfile_2resume'])


#     kick everything off
    trainer.run(dataloaders['train'], max_epochs=config['epochs'])

    wandb_logger_ins.close()


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


def remove_train_data_from_dataset(audio_dataset, config):
    print(config['loc_per_set'])
    keys2remove = []
    for key, sound_ins in audio_dataset.items():
        set_type = get_set_type(sound_ins.region, sound_ins.location,
                                config['loc_per_set'])
        if set_type == 'train':
            keys2remove.append(key)
    for key in keys2remove:
        del audio_dataset[key]
    return audio_dataset


def shorten_dataset4debug(audio_dataset,
                          config,
                          counts=(100, 10, 10),
                          ignore_train=False):
    train, test, val = [], [], []
    for key, sound_ins in audio_dataset.items():
        set_type = get_set_type(sound_ins.region, sound_ins.location,
                                config['loc_per_set'])
        if set_type == 'train':
            train.append(key)
        elif set_type == 'test':
            test.append(key)
        elif set_type == 'valid':
            val.append(key)
        else:
            train.append(key)

    # randomly pick 100 samples from train, 10 from test and 10 from val
    if ignore_train:
        train = np.empty(0)
    else:
        train = np.random.choice(train, counts[0], replace=False)
    test = np.random.choice(test, counts[1], replace=False)
    val = np.random.choice(val, counts[2], replace=False)
    # combine them
    debug_keys = train.tolist() + test.tolist() + val.tolist()
    all_keys = list(audio_dataset.keys())
    for key in all_keys:
        if key not in debug_keys:
            del audio_dataset[key]

    return audio_dataset


def make_locs_caseinsensitive(locs_per_set):
    # make all keys and values lower case
    # location_id_info is a dict with keys as set names and values as list of tuples
    # each tuple is (region, location_id)
    locs_per_set_lower = {}
    for key, val in locs_per_set.items():
        locs_per_set_lower[key.lower()] = [
            (x[0].lower(), x[1].lower()) for x in val
        ]

    # give a warning if there are duplicate keys
    if len(locs_per_set_lower.keys()) != len(locs_per_set.keys()):
        raise ValueError('Duplicate keys in location_id_info')
    # give a warning if there are duplicate values
    all_values = []
    for key, val in locs_per_set_lower.items():
        all_values.extend(val)
    if len(all_values) != len(set(all_values)):
        raise ValueError('Duplicate values in location_id_info')

    return locs_per_set_lower


def get_set_type(reg, loc, loc_per_set):
    """
    Given a tuple of (reg, location), returns 'train', 'valid', or 'test' depending on which one the location is in.
    """
    reg, loc = str(reg), str(loc)
    reg_loc_tuple = (reg.lower(), loc.lower())

    for set_lists in loc_per_set.values():
        if set_lists and not isinstance(set_lists[0], tuple):
            loc_per_set = make_locs_caseinsensitive(loc_per_set)

    for set_type in ['train', 'test', 'valid']:
        if reg_loc_tuple in loc_per_set[set_type]:
            return set_type

    print('WARNING: ' +
          f'This sample is NOT from a location ({reg_loc_tuple}) that is from' +
          ' pre-determined training,test,validation locations')
    return None


def split_train_test_val(x_data,
                         location_id_info,
                         onehot_encoded,
                         loc_per_set,
                         data_by_reference=False):
    X_train, X_test, X_val, y_train, y_test, y_val = [], [], [], [], [], []
    loc_id_train = []
    loc_id_test = []
    loc_id_valid = []
    assert len(x_data) == len(onehot_encoded)
    assert len(x_data) == len(location_id_info)
    loc_per_set = make_locs_caseinsensitive(loc_per_set)

    for sample, y_val_ins, loc_id in zip(  # type: ignore
            x_data, onehot_encoded, location_id_info):
        set_type = get_set_type(loc_id[0], loc_id[1], loc_per_set)
        if set_type == 'train':
            X_train.append(sample)
            y_train.append(y_val_ins)
            loc_id_train.append(loc_id)
        elif set_type == 'test':
            X_test.append(sample)
            y_test.append(y_val_ins)
            loc_id_test.append(loc_id)
        elif set_type == 'valid':
            X_val.append(sample)
            y_val.append(y_val_ins)
            loc_id_valid.append(loc_id)
        else:
            pass
    if not data_by_reference:
        if type(X_train[0]) == np.ndarray:
            X_train, X_test, X_val = np.array(X_train), np.array(
                X_test), np.array(X_val)

            X_train, X_test, X_val = torch.from_numpy(
                X_train).float(), torch.from_numpy(
                    X_test).float(), torch.from_numpy(X_val).float()
        elif type(X_train[0]) == torch.Tensor:
            # X_train, X_test, X_val = torch.stack(X_train), torch.stack(
            # X_test), torch.stack(X_val)
            pass
        else:
            raise ValueError(
                f'{X_train[0]} is neither a numpy array nor a torch tensor')

    # else:
    #     # we can just use reference to the audio instances
    #     pass

    y_train, y_test, y_val = np.array(y_train), np.array(y_test), np.array(
        y_val)
    y_train, y_test, y_val = torch.from_numpy(y_train).float(
    ), torch.from_numpy(y_test).float(), torch.from_numpy(y_val).float()

    return X_train, X_test, X_val, y_train, y_test, y_val


def setup(config):

    (device_str, exp_dir, run_id_2resume,
     checkpointfile_2resume) = (config['device'], config['exp_dir'],
                                config['run_id_2resume'],
                                config['checkpointfile_2resume'])

    if config['debug']:
        # print some debug info
        print('################---DEBUG MODE---###################')
        config['epochs'] = 5

    random_seed: int = 423590
    # reproducibility results
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.benchmark = False  # type: ignore

    Path(exp_dir).mkdir(exist_ok=True, parents=True)
    os.chdir(exp_dir)

    device = torch.device(f"cuda:{device_str}" if torch.cuda.is_available() else
                          "cpu")  # type: ignore
    config['device'] = device
    # wandb.init(config=config, project=wandb_project_name) # type: ignore
    # config = wandb.config # type: ignore

    if run_id_2resume == '':
        run_id = wandb.util.generate_id()  # type: ignore
    else:
        run_id = run_id_2resume
        print(f'run id found to be RESUMED!: {run_id}')

    if run_id_2resume != '' and run_id_2resume == '':
        raise Exception(
            'We need both run_id_2resume and checkpointfile_2resume to resume')
    elif run_id_2resume == '' and checkpointfile_2resume != '':
        raise Exception(
            'We need both run_id_2resume and checkpointfile_2resume to resume')
    config['run_id'] = run_id
    return config


def print_dataset_sizes(sound_datasets, y_train, y_test, y_val):
    print('train_size', len(sound_datasets['train']))
    if 'test' in sound_datasets:
        print('test_size', len(sound_datasets['test']))
    print('val_size', len(sound_datasets['val']))

    print('train category sizes:', torch.sum(y_train, 0))
    if 'test' in sound_datasets:
        print('test category sizes:', torch.sum(y_test, 0))
    print('val category sizes:', torch.sum(y_val, 0))


def put_samples_into_array(audio_dataset,
                           data_by_reference=False,
                           target_taxo=None,
                           y_type='binary'):  # sound_ins[1].taxo_code
    x_data = []
    y = []
    location_info = []
    for sound_ins in audio_dataset.values():
        if y_type == 'binary':
            sample_y_filtered = [
                taxo_code for taxo_code, y_value in sound_ins.taxo_y.items()
                if taxo_code in target_taxo and y_value == 1
            ]
        elif y_type == 'continuous':
            sample_y_filtered = [
                y_value for taxo_code, y_value in sound_ins.taxo_y.items()
                if taxo_code in target_taxo
            ]
        else:
            raise ValueError('y_type must be binary or continuous')
        # if data is in mmemory, we need to copy the data
        if not data_by_reference:
            if sound_ins.samples:
                raise Exception("samples deprecated! why is not it empty?")
            else:
                y.append(sample_y_filtered)
                x_data.append(sound_ins.data)
        # if loading data from disk every time, we can just use reference to the audio instances
        else:
            y.append(sample_y_filtered)
            x_data.append(sound_ins)

        location_info.append((sound_ins.region, sound_ins.location))

    return x_data, y, location_info
