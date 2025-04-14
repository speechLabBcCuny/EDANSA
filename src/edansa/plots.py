"""Plotting functions for the edansa package."""

import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import (auc, classification_report, confusion_matrix,
                             precision_recall_curve, roc_curve,
                             multilabel_confusion_matrix)
from sklearn.metrics import f1_score
from sklearn.preprocessing import label_binarize


def load_predictions(predict_file):
    """Load the predictions from a specified file."""
    print('MOVED TO wandbutils.py')
    return pd.read_csv(predict_file)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def apply_sig_on_predictions_df(predictions):
    # apply sigmoid to the predictions, if column name has 'pred' in it
    for col in predictions.columns:
        if 'pred' in col:
            predictions[col] = sigmoid(predictions[col])


def remove_outliers(df, column_name, multiplier=1.5):
    """Remove outliers using IQR method."""
    q1 = df[column_name].quantile(0.25)
    q3 = df[column_name].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (multiplier * iqr)
    upper_bound = q3 + (multiplier * iqr)

    new_df = df[(df[column_name] >= lower_bound) &
                (df[column_name] <= upper_bound)]
    num_removed = len(df) - len(new_df)
    return new_df, num_removed


def split_data(df):

    def filter_set(set_name):
        return df[df['set'] == set_name]

    data_sets = {
        'train': filter_set('train'),
        'val': filter_set('val'),
        'test': filter_set('test')
    }

    return data_sets


def remove_outliers_wrap(df, col_types2remove, multiplier=1.5):
    total_removed_data = 0
    # col_type list of  'target' and/or 'pred'
    for col_type in col_types2remove:
        # there should be only one column with the col_type
        #  (target_col_name or pred_col_name)
        for col_name in df.columns:
            if col_type in col_name:
                # print(f'Removing outliers from {col_name}...{col_type}')
                df, removed = remove_outliers(df,
                                              col_name,
                                              multiplier=multiplier)
                # print(f'Removed {removed} rows from {col_name}...{col_type}')
                total_removed_data += removed
                # print(f'Removed {removed} rows from {col_name}...{col_type}')

    return df, total_removed_data


def plot_ideal_line(all_data, target_col_name, ax):
    min_val, max_val = all_data[target_col_name].min(
    ), all_data[target_col_name].max()
    ax.plot([min_val, max_val], [min_val, max_val],
            color='r',
            label='Ideal Prediction')


def plot_predictions(data_sets,
                     target_col_name,
                     pred_col_name,
                     ax,
                     ideal_line=True,
                     title=None):

    all_data = pd.concat(data_sets.values())
    for label, data in data_sets.items():
        ax.scatter(data[target_col_name],
                   data[pred_col_name],
                   label=label.capitalize())

    ax.set_xlabel('True Labels')
    ax.set_ylabel('Predicted Labels')
    if ideal_line:
        plot_ideal_line(all_data, target_col_name, ax)
    ax.legend(loc='upper right')
    ax.set_title(title)


def filter_epochs2plot(epoch_metrics, epochs2plot, epochs2ignore):
    if epochs2plot and epochs2plot is not None:
        epoch_metrics = {
            epoch_metric.epoch: epoch_metric
            for epoch_metric in epoch_metrics.values()
            if epoch_metric.epoch in epochs2plot
        }

    if epochs2ignore and epochs2ignore is not None:
        epoch_metrics = {
            epoch_metric.epoch: epoch_metric
            for epoch_metric in epoch_metrics.values()
            if epoch_metric.epoch not in epochs2ignore
        }

    return epoch_metrics


def create_subplots(num_epochs):
    num_rows = (num_epochs + 2) // 3
    if num_rows == 0:
        num_rows = 1
    return plt.subplots(num_rows, 3, figsize=(25, 5 * num_rows))


def calculate_removed_data(original_predicts_len, data_sets):
    return original_predicts_len - sum(len(data) for data in data_sets.values())


def get_subplot_indexes(epoch_index):
    return epoch_index // 3, epoch_index % 3


def get_plot_title(epoch, col_types2remove, avg_removed_data):
    if col_types2remove:
        remove_outliers_text = ','.join(col_types2remove)
        title = (f'Removed outliers from {remove_outliers_text}:' +
                 f' {avg_removed_data:.2f}% of total. (Epoch:{epoch})')
    else:
        title = f'All data (Epoch {epoch})'
    return title


def prepare_data(predicts, col_types2remove, epoch):
    original_predicts_len = len(predicts)
    predicts, _ = remove_outliers_wrap(predicts, col_types2remove)
    data_sets = split_data(predicts)
    total_removed_data = calculate_removed_data(original_predicts_len,
                                                data_sets)
    avg_removed_data = total_removed_data / len(predicts) * 100
    title = get_plot_title(epoch, col_types2remove, avg_removed_data)
    return data_sets, title


# moved to wandbutils
# def get_files_to_plot(run_folder, project_name, run_file_name):


def plot_predictions_for_all_epochs(epoch_metrics,
                                    label_name,
                                    epochs2ignore=None,
                                    epochs2plot=None,
                                    every_n_epochs=-1,
                                    col_types2remove=None,
                                    ideal_line=True):
    epoch_metrics_c = epoch_metrics.copy()
    target_col_name, pred_col_name = (f'target_{label_name}',
                                      f'pred_{label_name}')

    if every_n_epochs > 0:
        epoch_metrics_c = {
            epoch_metric: epoch_metric
            for _, epoch_metric in epoch_metrics_c.items()
            if epoch_metric.epoch % every_n_epochs == 0
        }

    epoch_metrics_c = filter_epochs2plot(
        epoch_metrics_c,
        epochs2plot,
        epochs2ignore,
    )

    num_epochs = len(epoch_metrics_c)
    fig, axs = create_subplots(num_epochs)
    for metric_index, epoch_metric in enumerate(epoch_metrics_c.values()):
        epoch = epoch_metric.epoch
        predictions = epoch_metric.load_predictions()
        data_sets, title = prepare_data(predictions, col_types2remove, epoch)
        row_index, col_index = get_subplot_indexes(metric_index)

        plot_predictions(
            data_sets,
            target_col_name,
            pred_col_name,
            axs[row_index, col_index],  # type: ignore
            ideal_line=ideal_line,
            title=title)

    fig.tight_layout()
    plt.show()


def plot_predictions_for_all_configs(epoch_metrics, label_name, epoch_to_plot,
                                     configs):

    target_col_name, pred_col_name = (f'target_{label_name}',
                                      f'pred_{label_name}')

    fig, axs = plt.subplots(len(configs) // 2, 2, figsize=(18, 24))
    predictions = epoch_metrics[epoch_to_plot].load_predictions()
    # predict_file = predict_files[epoch_indexes.index(epoch_to_plot)]
    for i, (col_types2remove, ideal_line) in enumerate(configs):
        data_sets, title = prepare_data(predictions.copy(), col_types2remove,
                                        epoch_to_plot)
        row_index, col_index = i // 2, i % 2

        plot_predictions(
            data_sets,
            target_col_name,
            pred_col_name,
            axs[row_index, col_index],  # type: ignore
            ideal_line=ideal_line,
            title=title)

    fig.tight_layout()
    plt.show()
    return epoch_metrics[epoch_to_plot].predict_file


def calculate_bins(df):
    # calculate the interquartile range (IQR) of the data
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1

    # calculate the number of bins using the Freedman-Diaconis rule
    n = len(df)
    bin_width = 2 * iqr / np.power(n, 1 / 3)
    if bin_width > 0:
        num_bins = int(np.ceil((df.max() - df.min()) / bin_width))
    else:
        num_bins = 1

    return num_bins


def plot_dataset_col_dist(df,
                          plot_type='violin',
                          num_cols=2,
                          num_bins=None,
                          suptitle=None,
                          remove_outlier_multiplier=0.0):

    if remove_outlier_multiplier != 0.0:
        columns2process = df.columns
        df, _ = remove_outliers_wrap(df,
                                     columns2process,
                                     multiplier=remove_outlier_multiplier)

    # calculate the number of rows and columns needed to display all subplots
    num_vars = len(df.columns)
    num_rows = num_vars // num_cols + num_vars % num_cols
    fig_width = 8 * num_cols
    fig_height = 4 * num_rows
    fig_size = (fig_width, fig_height)

    # create a grid of subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=fig_size)
    axes = axes.flatten()  # type: ignore

    # create a plot for each column on a separate axis
    for i, col in enumerate(df.columns):
        data = df[[col]]
        # if 'rain_precip' in col:
        #     data = data[data < 0.1]
        # else:
        #     continue
        if plot_type == 'violin':
            sns.violinplot(x=col, data=data, ax=axes[i])
        elif plot_type == 'box':
            sns.boxplot(x=col, data=data, ax=axes[i])
        elif plot_type == 'kde':
            sns.kdeplot(data=data, ax=axes[i])
        elif plot_type == 'hist':
            # calculate the best number of bins for the histogram
            if num_bins is None:
                num_bins_col = calculate_bins(data[col])
            else:
                num_bins_col = num_bins
            axes[i].hist(data[col], bins=num_bins_col)
            axes[i].set_xlabel(col, fontsize=10)
            axes[i].set_ylabel('Frequency', fontsize=10)
            axes[i].tick_params(axis='both', which='major', labelsize=8)
            axes[i].tick_params(axis='both', which='minor', labelsize=8)
            axes[i].set_title(col, fontsize=10)
        axes[i].set_title(col)
    fig.subplots_adjust(wspace=0.3, hspace=0.5)
    # add title to the big figure
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=16)
    else:
        fig.suptitle('Distribution of Dataset Columns', fontsize=16)

    # hide the unused subplots
    for i in range(num_vars, num_rows * num_cols):
        fig.delaxes(axes[i])

    plt.show()


def calculate_class_report(predictions, class_names, thresholds):
    true_labels, pred_labels = get_true_pred(predictions, class_names)

    for i, class_name in enumerate(class_names):
        pred_labels[:, i] = (pred_labels[:, i]
                             > thresholds[class_name]).astype(int)

    if len(class_names) == 1:
        target_names = ['not ' + class_names[0], class_names[0]]
    else:
        target_names = class_names
    # print(target_names, true_labels, pred_labels)
    # Compute the classification report
    report = classification_report(true_labels,
                                   pred_labels,
                                   target_names=target_names,
                                   output_dict=True)
    # multi_conf_matrix = multilabel_confusion_matrix(true_labels, pred_labels)

    return report


def get_classification_report(predictions,
                              class_names,
                              thresholds,
                              title_addon=None,
                              plot=True):

    df_report = calculate_class_report(predictions, class_names, thresholds)

    # Plot the heatmap of the classification report
    if plot:
        plot_classification_report(df_report, title_addon=title_addon)

    return df_report


def plot_classification_report(report, title_addon=None):
    # Create a DataFrame from the classification report
    df_report = pd.DataFrame(report).transpose()

    # Plot the heatmap of the classification report
    _, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df_report.iloc[:-3, :-1],
                annot=True,
                cmap='Blues',
                cbar=True,
                ax=ax)
    ax.set_xlabel('Precision, Recall, F1-score')
    ax.set_ylabel('Class')
    if title_addon is None:
        title = 'Classification Report'
    else:
        title = f'Classification Report for {title_addon}'
    ax.set_title(title)
    plt.show()


def find_best_f1_epoch(
    predict_files,
    class_names,
    label_name='high',
):
    # deprecated
    raise DeprecationWarning('use wandbutils.get_metric_best_threshold')


def get_roc_curve(predictions,
                  class_names,
                  title_addon=None,
                  pred_prefix='pred_',
                  target_prefix='target_'):
    # Define the true labels and predicted labels
    true_labels, pred_scores = get_true_pred(predictions,
                                             class_names=class_names,
                                             pred_prefix=pred_prefix,
                                             target_prefix=target_prefix)

    true_labels_bin = true_labels
    # Compute the ROC Curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # for i in range(len(class_names)):
    for i, class_name in enumerate(class_names):
        fpr[class_name], tpr[class_name], _ = roc_curve(
            true_labels_bin[:, i],  # type: ignore
            pred_scores[:, i])
        roc_auc[class_name] = auc(fpr[class_name], tpr[class_name])

    # Compute the micro-average ROC Curve and AUC score
    fpr_micro, tpr_micro, _ = roc_curve(true_labels_bin.ravel(),
                                        pred_scores.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)

    # Plot the ROC Curve
    _, ax = plt.subplots(figsize=(8, 6))
    for _, class_name in enumerate(class_names):
        ax.plot(fpr[class_name],
                tpr[class_name],
                label=f'{class_name} (AUC = {roc_auc[class_name]:.2f})')
    ax.plot(fpr_micro,
            tpr_micro,
            label=f'Micro-average (AUC = {roc_auc_micro:.2f})',
            linestyle='--')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    if title_addon is None:
        title = 'ROC Curve'
    else:
        title = f'ROC Curve for {title_addon}'
    ax.set_title(title)
    ax.legend(loc='best')
    plt.show()
    return fpr, tpr, roc_auc


def get_class_names(class_names=None, prefix='rain_precip_'):
    if class_names is None:
        class_names = ['dry', 'low', 'medium', 'high']
    # Define the true labels and predicted labels
    # class_names_w_prefix = [prefix + i for i in class_names]

    return class_names


def get_true_pred(predictions,
                  class_names=None,
                  pred_prefix='pred_',
                  target_prefix='target_'):
    if class_names is None:
        class_names = ['dry', 'low', 'medium', 'high']
    # class_names_w_prefix = get_class_names(class_names)
    # Define the true labels and predicted labels
    target_class_names = [target_prefix + i for i in class_names]
    pred_class_names = [pred_prefix + i for i in class_names]

    true_labels = predictions[target_class_names].values
    pred_labels = predictions[pred_class_names].values

    return true_labels, pred_labels


def plot_confidence_vs_ground_truth(predictions_filtered,
                                    class_names=None,
                                    epoch=None,
                                    title_addon=None):
    del epoch

    labels = get_class_names(class_names)

    subset_df = predictions_filtered.loc[:,
                                         predictions_filtered.columns.str.
                                         startswith(
                                             ('target_', 'pred_', 'set'))]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    palette = {
        'Negative':
            (0.9333333333333333, 0.5215686274509804, 0.2901960784313726),
        'Positive':
            (0.2823529411764706, 0.47058823529411764, 0.8156862745098039),
    }

    for _, label in enumerate(labels):
        temp_df = pd.DataFrame()
        temp_df['confidence'] = subset_df[f'pred_{label}']
        temp_df['ground_truth'] = subset_df[f'target_{label}'].replace({
            1: 'Positive',
            0: 'Negative'
        })
        temp_df['category'] = label

        sns.stripplot(
            data=temp_df,
            x='confidence',
            y='category',
            hue='ground_truth',
            jitter=True,
            dodge=True,
            size=6,
            alpha=0.15,
            palette=palette,
            legend=False,  # type: ignore
        )

    handles = [
        plt.Line2D(  # type: ignore
            [0],
            [0],  # type: ignore
            marker='o',
            color=palette['Positive'],
            linestyle='',
            label='Positive',
            markersize=8),
        plt.Line2D(  # type: ignore
            [0],
            [0],  # type: ignore
            marker='o',
            color=palette['Negative'],
            linestyle='',
            label='Negative',
            markersize=8),
    ]

    ax1.legend(handles=handles, title='Ground Truth', loc='upper right')
    # title with epoch number
    if title_addon is None:
        title = 'Confidence vs Ground Truth'
    else:
        title = f'Confidence vs Ground Truth for {title_addon}'
    ax1.set(title=title, xlabel='Confidence', ylabel='Category')
    sns.despine(fig)
    plt.show()


def get_confusion_matrix(predictions,
                         class_names,
                         thresholds,
                         normalize=False,
                         title_addon=None,
                         plot=True,
                         pred_prefix='pred_',
                         target_prefix='target_'):
    # Define the true labels and predicted labels
    true_labels, pred_labels = get_true_pred(predictions,
                                             class_names,
                                             pred_prefix=pred_prefix,
                                             target_prefix=target_prefix)
    # true_labels = np.argmax(true_labels, axis=1)
    if isinstance(thresholds, pd.DataFrame):
        thresholds = dict(zip(thresholds.label_name, thresholds.threshold))

    # pred_labels = np.argmax(pred_labels, axis=1)
    # Use threshold to decide the class
    for i, class_name in enumerate(class_names):
        pred_labels[:, i] = (pred_labels[:, i]
                             > thresholds[class_name]).astype(int)

    # Compute the confusion matrix
    cms = multilabel_confusion_matrix(true_labels, pred_labels)
    cms_dict = {}
    print(cms)
    if len(cms) != len(class_names):
        print(
            f'WARNING: len(cms) != len(class_names): {len(cms)} != {len(class_names)}'
        )
        if len(class_names) == 1:
            print('adding one for not LABEL')
            class_names = ['not ' + class_names[0], class_names[0]]
    # if len(class_names) == 1:
    # class_names = ['not ' + class_names[0], class_names[0]]
    for i, (cm, class_name) in enumerate(zip(cms[::], class_names)):
        cms_dict[class_name] = cm
    if not plot:
        return cms_dict

    for i, (cm, class_name) in enumerate(zip(cms[::], class_names)):
        # Normalize the confusion matrix if desired
        if normalize:
            row_sums = cm.sum(axis=1, keepdims=True)
            nonzero_rows = np.nonzero(row_sums)[0]
            cm[nonzero_rows] = cm[nonzero_rows] / row_sums[nonzero_rows]
            cm = cm.round(2)
            fmt = '.2f'
        else:
            fmt = 'd'

        # Create a heatmap visualization of the confusion matrix
        _, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm,
                    annot=True,
                    fmt=fmt,
                    cmap='Blues',
                    xticklabels=[f'Not {class_name}', class_name],
                    yticklabels=[f'Not {class_name}', class_name])
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        if title_addon is None:
            title = f'Confusion Matrix for {class_name}'
        else:
            title = f'Conf. Matrix for {title_addon}, {class_name}'
        ax.set_title(title)
        plt.show()
    return cms_dict


def get_precision_recall_curve(predictions,
                               class_names,
                               title_addon=None,
                               plot=True):

    true_labels, pred_scores = get_true_pred(predictions,
                                             class_names=class_names)

    true_labels_bin = true_labels
    # Compute the precision, recall, and threshold values for each class
    precision = dict()
    recall = dict()
    thresholds = dict()
    pr_auc = {}
    for i, class_name in enumerate(class_names):
        precision[i], recall[i], thresholds[i] = precision_recall_curve(
            true_labels_bin[:, i], pred_scores[:, i])  # type: ignore
        pr_auc[class_name] = auc(recall[i], precision[i])

    # Compute the micro-average precision and recall scores
    precision_micro, recall_micro, _ = precision_recall_curve(
        true_labels_bin.ravel(), pred_scores.ravel())

    if not plot:
        return precision, recall, thresholds, pr_auc
    # Plot the Precision-Recall Curves
    _, ax = plt.subplots(figsize=(8, 6))
    for i in range(len(class_names)):
        ax.plot(recall[i], precision[i], label=f'{class_names[i]}')
    ax.plot(recall_micro,
            precision_micro,
            label='Micro-average',
            linestyle='--')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    if title_addon is None:
        title = 'Prec-Recall Curve'
    else:
        title = f'Prec-Recall Curve for {title_addon}'
    ax.set_title(title)
    ax.legend(loc='best')
    plt.show()

    return precision, recall, thresholds, pr_auc


def get_best_epoch(run, success_type='validation/loss'):

    # get all logged data using scan_history
    history = list(run.scan_history())
    if success_type in [
            'best_mean_ROC_AUC', 'best_min_ROC_AUC', 'val_roc_auc_mean'
    ]:
        _, best_epoch = get_best_value(max, success_type, history)
    elif success_type == 'validation/loss':
        _, best_epoch = get_best_value(min, success_type, history)
    else:
        raise ValueError('success_type not defined')

    return best_epoch


def get_best_value(selection_func, success_type, history):
    validation_loss_values = [row for row in history if success_type in row]
    # get the lowest validation loss
    best_row = selection_func(validation_loss_values,
                              key=lambda x: x[success_type])
    return best_row, best_row['epoch']


def get_run_checkpoints_folder(run_folder, project_name, run_id):
    run_files = glob.glob(f'{run_folder}{project_name}/checkpoints/*{run_id}*')
    if len(run_files) == 0:
        print(f'No run file found for {run_id} at {run_folder}')
        return None
    assert len(run_files) == 1, 'There should be only one run file'
    return run_files[0]


def get_f1_threshold(
        predictions,
        class_names,
        label_name,
        SAVE=False,  # pylint: disable=invalid-name
        output_file_name=None,
        eval_func=f1_score):

    val_set = predictions[predictions['set'] == 'val'].copy()

    # calculate threshold for rain_precip_high and rain_precip_dry
    thresholds = {}
    f1_scores = {}
    for class_name in class_names:
        # Iterate through a range of thresholds to find the best one
        best_threshold = 0
        best_f1_score = 0
        for threshold in np.arange(0, 1, 0.01):
            val_set['pred_label'] = (val_set[f'pred_{label_name}_{class_name}']
                                     > threshold).astype(int).copy()
            f1 = eval_func(val_set[f'target_{label_name}_{class_name}'],
                           val_set['pred_label'])
            if f1 > best_f1_score:
                best_f1_score = f1
                best_threshold = threshold
        thresholds[f'{label_name}_{class_name}'] = best_threshold
        f1_scores[f'{label_name}_{class_name}'] = best_f1_score
        print(f'Best threshold for {class_name}:', best_threshold)
        print(f'Best F1 score for {class_name}:', best_f1_score)

    thresholds_f1_df = pd.DataFrame({
        'category': thresholds.keys(),
        'threshold': list(thresholds.values()),
        'f1_score': list(f1_scores.values())
    })
    if SAVE:
        assert output_file_name is not None, 'output_file_name is None'

        thresholds_f1_df.to_csv(output_file_name, index=False)
    return thresholds_f1_df


def load_neon_data(tool_weather_data_path,
                   length=5,
                   location='',
                   region='',
                   local_time_zone='America/Anchorage'):
    neon_files = glob.glob(f'{tool_weather_data_path}/*TOOL*/*{length}min*.csv')
    neon_df = []
    for file in neon_files:
        w_d = pd.read_csv(file)
        w_d['startDateTime'] = pd.to_datetime(
            w_d['startDateTime']).dt.tz_convert(local_time_zone).dt.tz_localize(
                None)
        w_d['endDateTime'] = pd.to_datetime(w_d['endDateTime']).dt.tz_convert(
            local_time_zone).dt.tz_localize(None)
        if location != '':
            w_d['location'] = location.lower()
        if region != '':
            w_d['region'] = region.lower()

        w_d['TIMESTAMP'] = w_d['startDateTime']
        neon_df.append(w_d)
    # priPrecipFinalQF is the quality flag for precipitation
    # 0 is good, 1 is bad
    neon_df = pd.concat(neon_df)
    neon_df = neon_df.sort_values(by=['startDateTime'])
    neon_df = neon_df.reset_index(drop=True)

    return neon_df
