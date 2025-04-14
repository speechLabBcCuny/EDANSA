from ignite.metrics import EpochMetric
import numpy as np


def activated_output_transform(output):
    y_pred, y = output
    #     y_pred = torch.exp(y_pred)
    return y_pred, y


def roc_auc_per_class_compute_fn(y_preds, y_targets):
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError as exc:
        raise RuntimeError(
            'This contrib module requires sklearn to be installed.') from exc

    y_true = y_targets.numpy()
    y_pred = y_preds.numpy()

    # Check for NaN values first
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        nan_y_true_count = np.isnan(y_true).sum()
        nan_y_pred_count = np.isnan(y_pred).sum()
        print("Warning: NaN values detected.")
        if nan_y_true_count > 0:
            print(f"NaN values detected in y_true: {nan_y_true_count}")
        if nan_y_pred_count > 0:
            print(f"NaN values detected in y_pred: {nan_y_pred_count}")
        # Determine return shape based on input shape
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            num_classes = y_true.shape[1]
            return np.array([0.5] * num_classes)
        else:
            return np.array([0.5])  # Return array for binary/ambiguous case

    # Handle binary classification (1D input or 2D with one column)
    if len(y_true.shape) == 1 or y_true.shape[1] == 1:
        # Ensure y_true and y_pred are 1D for roc_auc_score
        y_true_flat = y_true.ravel()
        y_pred_flat = y_pred.ravel()
        # Check if y_true contains only one class
        if len(np.unique(y_true_flat)) <= 1:
            return np.array([0.5])  # AUC is not defined, return array [0.5]
        else:
            # Calculate AUC for binary case using average=None and return as array
            # roc_auc_score with average=None should return an array even for binary
            return np.array(
                roc_auc_score(y_true_flat, y_pred_flat, average=None))

    # Handle multi-class classification (2D input with multiple columns)
    else:
        num_classes = y_true.shape[1]
        scores = []
        for i in range(num_classes):
            # Check if y_true for class i contains only one class
            if len(np.unique(y_true[:, i])) <= 1:
                scores.append(0.5)  # AUC is not defined
            else:
                scores.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
        return np.array(scores)


#[docs]
class ROC_AUC_perClass(EpochMetric):
    """Computes Area Under the Receiver Operating Characteristic Curve (ROC AUC)
  accumulating predictions and the ground-truth during an epoch and applying
  `sklearn.metrics.roc_auc_score <http://scikit-learn.org/stable/modules/generated/
  sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score>`_ .

  Args:
      output_transform (callable, optional): a callable that is used to transform the
          :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
          form expected by the metric. This can be useful if, for example, you have a multi-output model and
          you want to compute the metric with respect to one of the outputs.
      check_compute_fn (bool): Optional default False. If True, `roc_curve
          <http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#
          sklearn.metrics.roc_auc_score>`_ is run on the first batch of data to ensure there are
          no issues. User will be warned in case there are any issues computing the function.

  ROC_AUC expects y to be comprised of 0's and 1's. y_pred must either be probability estimates or confidence
  values. To apply an activation to y_pred, use output_transform as shown below:

  .. code-block:: python

      def activated_output_transform(output):
          y_pred, y = output
          y_pred = torch.sigmoid(y_pred)
          return y_pred, y

      roc_auc = ROC_AUC(activated_output_transform)

  """

    def __init__(self,
                 output_transform=lambda x: x,
                 check_compute_fn: bool = False):
        #         print(output_transform)
        super(ROC_AUC_perClass,
              self).__init__(roc_auc_per_class_compute_fn,
                             output_transform=output_transform,
                             check_compute_fn=check_compute_fn)


def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype('int')


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class F1_Score_with_Optimal_Threshold(EpochMetric):

    def __init__(self,
                 output_transform=lambda x: x,
                 check_compute_fn: bool = False,
                 thresholds=np.arange(0, 1, 0.001)):
        self.thresholds = thresholds
        super(F1_Score_with_Optimal_Threshold,
              self).__init__(self._compute_fn_with_optimal_threshold,
                             output_transform=output_transform,
                             check_compute_fn=check_compute_fn)

    def _compute_fn_with_optimal_threshold(self, y_preds, y_targets):
        """
        Compute the F1 score for each class using the optimal threshold.

        Args:
            y_preds (torch.Tensor): Predicted probabilities of shape (batch_size, num_classes).
            y_targets (torch.Tensor): Target labels of shape (batch_size, num_classes).

        Returns:
            numpy.ndarray: F1 score for each class of shape (num_classes,).
        """
        try:
            from sklearn.metrics import f1_score
        except ImportError as exc:
            raise RuntimeError(
                "This contrib module requires sklearn to be installed."
            ) from exc
        y_true = y_targets.numpy()
        y_pred_probs = y_preds.numpy()
        y_pred_probs = sigmoid(y_pred_probs)
        if len(y_true.shape) == 1:
            y_true = y_true.reshape(-1, 1)
            y_pred_probs = y_pred_probs.reshape(-1, 1)
        n_classes = y_true.shape[1]
        scores = np.zeros((n_classes, len(self.thresholds)))
        for i in range(n_classes):
            y_true_i = y_true[:, i]
            y_pred_probs_i = y_pred_probs[:, i]
            for j, t in enumerate(self.thresholds):
                y_pred_i = to_labels(y_pred_probs_i, t)
                scores[i, j] = f1_score(y_true_i, y_pred_i, average='binary')

        # how to return best threshold for each class
        # best_thresholds = self.thresholds[np.argmax(scores, axis=1)]

        # then recalculate f1 score with best thresholds
        # y_pred = to_labels(y_pred_probs,
        #    np.tile(best_thresholds, (y_true.shape[0], 1)))

        # res = f1_score(y_true, y_pred, average=None)
        return np.max(scores, axis=1)
