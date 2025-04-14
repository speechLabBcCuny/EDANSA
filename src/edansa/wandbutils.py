"""Utility functions for wandb."""

import wandb
import glob
import numpy as np
import pandas as pd
from pathlib import Path
import inspect
# from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, mean_squared_error
import json


def get_wandb_config(run_identity,
                     config_file=None,
                     api=None,
                     updatewandb=False):
    """Get the wandb config for a run. If config_file is provided, it will be
    used to update the config. """
    run = None
    if not config_file or updatewandb:
        # authenticate wandb
        if api is None:
            wandb.login()
            api = wandb.Api()
        run = api.run(run_identity)
        config = run.config

    if config_file:
        print('Loading config file from {}'.format(config_file))
        with open(config_file, 'r', encoding='utf-8') as f:
            json_config = json.load(f)
        cleaned_config = clean_json_config(json_config)

        if updatewandb:
            config.update(cleaned_config)
        else:
            config = cleaned_config

    return config, run


def clean_json_config(json_config):
    cleaned_config = {}
    for key, value in json_config.items():
        if 'value' in value:
            cleaned_config[key] = value['value']
        else:
            cleaned_config[key] = value
    return cleaned_config


def use_default_value_if_none(func):

    def wrapper(self, *args, **kwargs):
        sig = inspect.signature(func)
        bound_args = sig.bind(self, *args, **kwargs)
        bound_args.apply_defaults()

        for param_name, value in bound_args.arguments.items():
            if value is None and param_name != 'self':
                default_attr_name = f'{param_name}'
                if hasattr(self, default_attr_name):
                    bound_args.arguments[param_name] = getattr(
                        self, default_attr_name)

        return func(*bound_args.args, **bound_args.kwargs)

    return wrapper


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class EpochMetric:
    """ Holds metrics and inference results for each epoch. """

    def __init__(
        self,
        epoch=None,
        predict_file=None,
        predictions=None,
        label_names=None,
        pred_suffix='pred_',
        target_suffix='target_',
    ) -> None:
        self.epoch = epoch
        self.predict_file = predict_file
        self.label_names = label_names
        self.predictions = predictions
        self.sigmoid_applyed = False
        self.pred_suffix = pred_suffix
        self.target_suffix = target_suffix
        if self.epoch is None:
            if self.predict_file is not None:
                self.epoch = self.get_epoch4predicts_file_name(predict_file)
            else:
                raise ValueError(
                    'Either epoch or predict_file must be specified.')

    def get_epoch4predicts_file_name(self, file_name):
        # print(file_name)
        # all_predictions_epoch_1.csv
        last_part = file_name.split('_')[-1]
        # print(a)
        last_part_noprefix = last_part.split('.')[0]
        # print(b)
        try:
            return int(last_part_noprefix)
        except ValueError:
            print(f'Epoch number could not be extracted from {file_name}.')

    # make class sortable by epoch
    def __lt__(self, other):
        return self.epoch < other.epoch

    def __repr__(self):
        return f'EpochMetrics(epoch={self.epoch})'

    def __str__(self):
        return f'EpochMetrics(epoch={self.epoch})'

    def __eq__(self, other):
        return self.epoch == other.epoch

    def __hash__(self):
        return hash(self.epoch)

    def apply_sig_on_predictions_df(self, predictions=None):
        if self.sigmoid_applyed:
            return self.predictions
        # apply sigmoid to the predictions, if column name has 'pred' in it
        predictions = (predictions
                       if predictions is not None else self.predictions)
        assert predictions is not None, 'predictions must be specified.'
        for col in predictions.columns:
            if 'pred' in col:
                predictions[col] = sigmoid(predictions[col])
        self.sigmoid_applyed = True
        return predictions

    def load_predictions(self, predict_file=None, apply_sigmoid=False):
        """Load the predictions from a specified file."""
        if self.predictions is not None:
            return self.predictions
        predict_file = predict_file or self.predict_file
        assert predict_file is not None, 'predict_file must be specified.'
        predictions = pd.read_csv(predict_file)
        if apply_sigmoid:
            predictions = self.apply_sig_on_predictions_df(predictions)
        self.predictions = predictions
        return self.predictions

    def get_f1_thresholds_filename(
        self,
        epoch,
        predict_file=None,
    ):
        """Get the filename for the thresholds and f1 scores."""
        predict_file = predict_file or self.predict_file
        assert predict_file is not None
        predict_file = Path(
            predict_file
        ).parent.parent / f'thresholds_f1_scores_epoch={epoch}.csv'
        file_name = str(predict_file).replace('wandb', 'checkpoints')
        return file_name

    def get_metric_best_threshold(
            self,
            predictions=None,
            label_names=None,
            SAVE=False,  # pylint: disable=invalid-name
            output_file_name=None,
            sets2use=('val',),
            eval_func=f1_score,
            range_thresholds=np.arange(0, 1.01, 0.01),
    ):
        """ Get the best threshold for each class based on the metric

        Args:
            predictions (pd.DataFrame): predictions dataframe
                requires columns: set, f'pred_{label_name}',
                 'target_{label_name}'
            class_names (list): list of class names
            label_name (str): label name
            SAVE (bool, optional): whether to save the thresholds.
                Defaults to False.
            output_file_name (str, optional): output file name. Defaults to None
            sets2use (list, optional): list of sets to use. Defaults to ['val'].
            eval_func (function, optional): evaluation function.
                Defaults to f1_score.

        """
        sets2use = list(sets2use)
        if predictions is None:
            predictions = self.load_predictions(apply_sigmoid=True)
        else:
            print('WARNING - predictions are passed as an argument,' +
                  'make sure sigmoid is already applied')
        if predictions is None:
            raise ValueError('predictions must be specified')
        label_names = label_names or self.label_names
        if label_names is None:
            raise ValueError('label_names must be specified')
        predsbyset = predictions[predictions['set'].isin(sets2use)].copy()
        # calculate threshold for rain_precip_high and rain_precip_dry
        thresholds = {}
        scores = {}
        for label_name in label_names:
            # Iterate through a range of thresholds to find the best one
            best_threshold = 0
            best_score = 0
            for threshold in range_thresholds:
                try:
                    predsbyset['pred_label'] = (
                        predsbyset[f'{self.pred_suffix}{label_name}']
                        >= threshold).astype(int).copy()
                except KeyError as e:
                    print(
                        'WARNING pred_{label_name} not found in predictions: ' +
                        f'{e}')
                f1 = eval_func(predsbyset[f'{self.target_suffix}{label_name}'],
                               predsbyset['pred_label'],
                               average='binary')
                if f1 > best_score:
                    best_score = f1
                    best_threshold = threshold
            thresholds[f'{label_name}'] = best_threshold
            scores[f'{label_name}'] = best_score

        info_df = pd.DataFrame({
            'label_name': thresholds.keys(),
            'threshold': list(thresholds.values()),
            eval_func.__name__: list(scores.values())
        })
        self.threshold_info = info_df
        if SAVE:
            assert output_file_name is not None, 'output_file_name is None'
            Path(output_file_name).parent.mkdir(parents=False, exist_ok=True)

            info_df.to_csv(output_file_name, index=False)
        return info_df

    def get_mse_loss(self, label_names=None, set2eval='val'):
        predictions = self.load_predictions(apply_sigmoid=False)
        assert predictions is not None, 'predictions must be specified'

        assert set2eval in ['val', 'valid',
                            'test'], 'set2eval must be val or test'
        eval_set = predictions[predictions['set'] == set2eval].copy()

        label_names = label_names or self.label_names
        assert label_names is not None, 'label_names must be specified'
        mse_baseline = {}
        for label in label_names:
            target_label = f'{self.target_suffix}{label}'
            pred_label = f'{self.pred_suffix}{label}'
            eval_set_label = self.ignore_nan_rows(
                label_name=target_label,
                df=eval_set,
            )
            res = mean_squared_error(eval_set_label[target_label],
                                     (eval_set_label[pred_label]))
            mse_baseline[label] = res
        return mse_baseline

    def ignore_nan_rows(
        self,
        label_name=None,
        df=None,
    ):
        assert df is not None, 'df must be specified'
        if label_name and label_name not in df.columns:
            print(f'WARNING: {label_name} not in df columns')
            return df.copy()
        if label_name:
            nan_count = df[[label_name]].isna().sum()
        else:
            nan_count = df.isna().sum()

        if nan_count.sum() > 0:
            print(
                f'WARNING: {nan_count.sum()} nan values in the dataframe for {label_name}'
            )
            print('Ignoring rows with nan values !!!!')
            if label_name:
                return df[df[label_name].notna()].copy()
            else:
                return df.dropna()
        return df.copy()

    def get_mse_baseline(self, label_names=None, set2eval='val'):
        '''
        Get the mse baseline for each label_name

        Args:
            label_names (list): list of label names
            set2eval (str, optional): set to use for evaluation.
        '''
        predictions = self.load_predictions(apply_sigmoid=False)
        assert predictions is not None, 'predictions must be specified'

        train = predictions[predictions['set'] == 'train'].copy()

        assert set2eval in ['val', 'valid',
                            'test'], 'set2eval must be val or test'
        eval_set = predictions[predictions['set'] == set2eval]

        label_names = label_names or self.label_names
        assert label_names is not None, 'label_names must be specified'
        mse_baseline = {}
        train_stats = {}
        for label in label_names:
            target_label = f'{self.target_suffix}{label}'
            pred_label = f'{self.pred_suffix}{label}'
            train_label = self.ignore_nan_rows(label_name=target_label,
                                               df=train)

            train_stats[label] = train_label[target_label].describe()
            threshold = train_stats[label]['mean']
            eval_set_label = eval_set[[target_label, pred_label]].copy()
            eval_set_label = self.ignore_nan_rows(
                label_name=target_label,
                df=eval_set_label,
            )

            print(f'threshold for {label}: {threshold}')
            print(
                f'len of eval set {len(eval_set_label)}, len of pred {len(eval_set_label[pred_label])}, len of target {len(eval_set_label[target_label])}'
            )
            res = mean_squared_error(
                eval_set_label[target_label],
                np.array([threshold] * len(eval_set_label[pred_label])))
            mse_baseline[label] = res
        return mse_baseline, train_stats
        # best = res
        # best_threshold = threshold


class WandbWrapped:
    """Wrapper for wandb to make it easier to use."""

    def __init__(
        self,
        api=None,
        wandb_user_name='',
        wandb_project_name='',
        local_project_name='',
        run_id='',
        run_folder='',
        run_time='',
        run_checkpoints_folder='',
        label_names='',
    ) -> None:
        if api is None:
            wandb.login()
            self.api = wandb.Api()
        else:
            self.api = api
        self.run_id = run_id
        self.wandb_user_name = wandb_user_name
        self.run_folder = run_folder
        self.local_project_name = local_project_name
        self.wandb_project_name = wandb_project_name
        self.run_time = run_time
        self.run_checkpoints_folder = run_checkpoints_folder
        self.label_names = label_names
        self.change_run()

    # @use_default_value_if_none
    def change_run(
        self,
        run_id=None,
        wandb_user_name=None,
        run_folder=None,
        local_project_name=None,
        wandb_project_name=None,
        run_time=None,
        label_names=None,
    ):
        self.run_id = run_id or self.run_id
        self.wandb_user_name = wandb_user_name or self.wandb_user_name
        self.run_folder = run_folder or self.run_folder
        self.local_project_name = local_project_name or self.local_project_name
        self.wandb_project_name = wandb_project_name or self.wandb_project_name
        self.label_names = label_names or self.label_names
        self.history = None
        if self.run_id:
            # here order is important
            self.run_time = run_time or self.get_run_time()
            self.run_handles()
            self.get_epoch_metrics()
            self.set_run()

    def set_run(self,):
        ff = f'{self.wandb_user_name}/{self.wandb_project_name}/{self.run_id}'
        try:
            self.run = self.api.run(ff)
        except:  # pylint: disable=bare-except
            print(f'Error:Run not found! at {ff}')

    def get_run_checkpoints_folder(self,
                                   run_folder=None,
                                   local_project_name=None,
                                   run_id=None):
        """Using run_id to find the folder with checkpoints of the run"""
        run_folder = run_folder or self.run_folder
        local_project_name = local_project_name or self.local_project_name
        run_id = run_id or self.run_id
        if run_id is None or not run_id:
            print('No run id provided')
            return None
        glob_path = f'{run_folder}{local_project_name}/checkpoints/*{run_id}*'
        run_files = glob.glob(glob_path)
        if len(run_files) == 0:
            print(f'No run file found for {run_id} at {glob_path}')
            return None
        assert len(run_files) == 1, 'There should be only one run file'
        self.run_checkpoints_folder = run_files[0]
        return self.run_checkpoints_folder

    def get_run_time(self, run_checkpoints_folder=None):
        run_checkpoints_folder = (run_checkpoints_folder or
                                  self.get_run_checkpoints_folder())
        if run_checkpoints_folder is None or not run_checkpoints_folder:
            print('No run checkpoints folder provided')
            return None
        run_time = run_checkpoints_folder.split('/')[-1].split('-')[1]
        self.run_time = run_time
        return run_time

    def run_handles(self,):
        wandb_folder_name = f'run-{self.run_time}-{self.run_id}'
        identity = (
            f'{self.wandb_user_name}/{self.wandb_project_name}/{self.run_id}')
        self.wandb_folder_name = wandb_folder_name
        self.identity = identity
        return self.wandb_folder_name, self.identity

    def get_epoch_metrics(self,
                          run_folder=None,
                          local_project_name=None,
                          wandb_folder_name=None):
        print(f'Getting epoch metrics for {self.run_id}')
        run_folder = run_folder or self.run_folder
        local_project_name = local_project_name or self.local_project_name
        wandb_folder_name = wandb_folder_name or self.wandb_folder_name

        predict_dir_path = (f'{run_folder}{local_project_name}' +
                            f'/wandb/{wandb_folder_name}/files/')
        predict_files = sorted(glob.glob(f'{predict_dir_path}/*epoch*.csv'))

        epoch_metrics = [
            EpochMetric(predict_file=file_name, label_names=self.label_names)
            for file_name in predict_files
        ]
        epoch_metrics = sorted(epoch_metrics)
        # make epoch_metrics a dict
        epoch_metrics = {
            epoch_metric.epoch: epoch_metric for epoch_metric in epoch_metrics
        }
        self.epoch_metrics = epoch_metrics
        return self.epoch_metrics

    def get_best_epoch(self, success_type='validation/loss', version=1):
        """Get the best epoch based on the success_type using wandb records.
        """

        def get_best_value(selection_func, success_type, history):
            validation_loss_values = [
                row for row in history if success_type in row
            ]
            # for some reason there are more values than epoch counts
            # filled with None
            validation_loss_values = [
                m for m in validation_loss_values if m[success_type] is not None
            ]
            # get the lowest validation loss
            best_row = selection_func(validation_loss_values,
                                      key=lambda x: x[success_type])
            return best_row, best_row['epoch']

        # get all logged data using scan_history
        self.history = list(
            self.run.scan_history(keys=[
                success_type, 'epoch'
            ], page_size=1e5)) if self.history is None else self.history
        print(f'lenght of history: {len(self.history)}')
        if success_type in [
                'best_mean_ROC_AUC',
                'best_min_ROC_AUC',
                'val_roc_auc_mean',
                'best_mean_f1',
                'best_min_f1',
        ]:
            best_row, best_epoch = get_best_value(max, success_type,
                                                  self.history)
        elif success_type in ['validation/loss', 'test/loss']:
            best_row, best_epoch = get_best_value(min, success_type,
                                                  self.history)
        else:
            raise ValueError('success_type not defined')
        # print(_)
        self.best_epoch = best_epoch
        self.best_epoch_type = success_type
        if version == 1:
            return self.best_epoch, self.best_epoch_type
        else:
            return self.best_epoch, self.best_epoch_type, best_row

    def get_best_f1_from_local_metrics(self,
                                       gather_func=np.min,
                                       label_names=None,
                                       sets2use=('val',)):
        """Get the best epoch based on the success_type using local metrics.

        Using EpochMetric's get_metric_best_threshold function.

        gather_func: function to use to gather the results from the epochs.
            ex: max, min, np.mean
        """
        label_names = label_names or self.label_names
        best_epoch = None
        best_score = 0
        for epoch, epoch_metric in self.epoch_metrics.items():
            info_df = epoch_metric.get_metric_best_threshold(
                label_names=label_names,
                SAVE=False,
                eval_func=f1_score,
                sets2use=sets2use,
            )
            if info_df is None:
                continue
            best_score_epoch = gather_func(info_df['f1_score'])
            if best_score_epoch > best_score:
                best_score = best_score_epoch
                best_epoch = epoch
        return best_epoch, best_score
