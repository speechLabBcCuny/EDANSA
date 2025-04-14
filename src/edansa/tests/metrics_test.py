import torch
import numpy as np
import pytest
from edansa.metrics import F1_Score_with_Optimal_Threshold, roc_auc_per_class_compute_fn


def test_f1_score_with_optimal_threshold():
    # Create a sample dataset
    y_true = torch.tensor([[0, 1], [1, 0], [1, 1], [0, 0]])
    y_pred = torch.tensor([[0.2, 0.8], [0.6, 0.4], [0.7, 0.3], [0.3, 0.7]])

    # Create an instance of the F1_Score_with_Optimal_Threshold metric
    f1_score = F1_Score_with_Optimal_Threshold()

    # Compute the F1 score
    f1_score.update((y_pred, y_true))
    f1 = f1_score.compute()
    # Check that the F1 score is correct
    assert f1.shape == (2,)

    assert abs(f1[0] - 1) < 1e-6
    assert abs(f1[1] - 0.66666667) < 1e-6

    # Test the metric with a one-dimensional input
    y_true = torch.tensor([1, 0, 1, 0])
    y_pred = torch.tensor([0.8, 0.4, 0.6, 0.2])
    f1_score = F1_Score_with_Optimal_Threshold()
    f1_score.update((y_pred, y_true))
    f1 = f1_score.compute()
    assert f1.shape == (1,)
    assert abs(f1 - 1) < 1e-6, f1

    # Test the metric with a different threshold range
    thresholds = [0.1, 0.3, 0.5, 0.9]
    f1_score = F1_Score_with_Optimal_Threshold(thresholds=thresholds)
    f1_score.update((y_pred, y_true))
    f1 = f1_score.compute()
    assert f1.shape == (1,)
    assert abs(f1 - 0.66666667) < 1e-6, f1

    # Test the metric with a different output transform
    output_transform = lambda x: (torch.sigmoid(x[0]), x[1])
    f1_score = F1_Score_with_Optimal_Threshold(
        output_transform=output_transform)
    f1_score.update((y_pred, y_true))
    f1 = f1_score.compute()
    assert f1.shape == (1,)
    assert abs(f1 - 1) < 1e-6, f1


def test_binary_classification_roc_auc():
    # Valid case
    y_true = torch.tensor([0, 1, 0])
    y_pred = torch.tensor([0.2, 0.8, 0.3], dtype=torch.float32)
    assert np.isclose(roc_auc_per_class_compute_fn(y_pred, y_true), 1.0)

    # Only one class in y_true
    y_true = torch.tensor([1, 1, 1])
    y_pred = torch.tensor([0.2, 0.8, 0.3], dtype=torch.float32)
    assert roc_auc_per_class_compute_fn(y_pred, y_true) == 0.5


def test_multiclass_classification_roc_auc():
    # Valid case
    y_true = torch.tensor([[1, 0], [0, 1], [1, 0], [0, 1]])
    y_pred = torch.tensor([[0.8, 0.2], [0.3, 0.7], [0.7, 0.3], [0.4, 0.6]],
                          dtype=torch.float32)
    result = roc_auc_per_class_compute_fn(y_pred, y_true)
    assert np.allclose(result, [1.0, 1.0])

    # Only one class for one of the labels
    y_true = torch.tensor([[1, 0], [1, 0], [1, 0], [1, 0]])
    y_pred = torch.tensor([[0.8, 0.2], [0.3, 0.7], [0.7, 0.3], [0.4, 0.6]],
                          dtype=torch.float32)
    result = roc_auc_per_class_compute_fn(y_pred, y_true)
    assert np.allclose(result, [0.5, 0.5])


def test_invalid_input_roc_auc():
    # If the length of y_true and y_pred does not match
    with pytest.raises(ValueError):
        y_true = torch.tensor([1, 0, 1])
        y_pred = torch.tensor([0.2, 0.8, 0.3, 0.7], dtype=torch.float32)
        roc_auc_per_class_compute_fn(y_pred, y_true)

    # If the shape of y_true and y_pred in multi-class scenario does not match
    with pytest.raises(ValueError):
        y_true = torch.tensor([[1, 0], [0, 1], [1, 0]])
        y_pred = torch.tensor([[0.8, 0.2], [0.3, 0.7], [0.7, 0.3], [0.4, 0.6]],
                              dtype=torch.float32)
        roc_auc_per_class_compute_fn(y_pred, y_true)
