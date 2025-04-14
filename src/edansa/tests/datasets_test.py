import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
import random
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset

from edansa.datasets import audioDataset, AugmentingAudioDataset

# Define devices to test on if GPU is available
DEVICES = [torch.device('cpu')]
if torch.cuda.is_available():
    DEVICES.append(torch.device('cuda:0'))

# --- Fixtures ---


@pytest.fixture(params=DEVICES)
def device(request):
    """Fixture to provide CPU and CUDA devices for testing."""
    return request.param


@pytest.fixture
def dummy_numpy_data(device):
    """Creates dummy numpy data for testing, returns tensors on device."""
    # Keep initial creation as numpy for testing that path
    X_np = np.random.rand(5, 1, 1600).astype(np.float32)
    y_np = np.random.randint(0, 2, (5, 3)).astype(np.float32)
    # Ensure at least one sample has silence (assuming index 0 is silence)
    y_np[0, 0] = 1
    y_np[1, 0] = 1
    # Convert to tensors on the target device for the fixture output
    X_tensor = torch.from_numpy(X_np).to(device)
    y_tensor = torch.from_numpy(y_np).to(device)
    return X_np, y_tensor  # Return original numpy X for testing loading path, and tensor y


@pytest.fixture
def dummy_list_data(device):
    """Creates dummy list data (numpy arrays) for testing, returns list and tensor y."""
    # Keep inner elements as numpy arrays for testing that path
    X_list_np = [np.random.rand(1, 1600).astype(np.float32) for _ in range(5)]
    y_np = np.random.randint(0, 2, (5, 3)).astype(np.float32)
    y_np[0, 0] = 1
    y_np[1, 0] = 1
    # Convert y to tensor on the target device
    y_tensor = torch.from_numpy(y_np).to(device)
    return X_list_np, y_tensor  # Return list of numpy X, tensor y


@pytest.fixture
def mock_data_reference(device):
    """Creates mock objects for data_by_reference testing."""
    mocks = []
    expected_data = []  # Store the numpy data the mocks will return
    for i in range(5):
        mock = MagicMock()
        # Ensure consistent shape, return numpy as per comments
        data_np = np.random.rand(1, 1600).astype(np.float32)
        mock.get_data_by_value.return_value = (data_np, 16000)
        mocks.append(mock)
        expected_data.append(data_np)
    # Create y tensor on device
    y_np = np.random.randint(0, 2, (5, 3)).astype(np.float32)
    y_np[0, 0] = 1
    y_np[1, 0] = 1
    y_tensor = torch.from_numpy(y_np).to(device)
    return mocks, y_tensor, expected_data


# --- Tests for audioDataset ---


def test_audioDataset_init_numpy(dummy_numpy_data, device):
    """Tests audioDataset initialization with numpy data."""
    X_np, y_tensor = dummy_numpy_data
    dataset = audioDataset(X_np, y_tensor, device=device)
    assert len(dataset) == 5
    assert dataset.X is X_np  # Should store the original numpy array
    assert torch.equal(dataset.y, y_tensor)  # y should be tensor on device
    assert dataset.transform is None
    assert not dataset.data_by_reference
    assert dataset.device == device


def test_audioDataset_init_list(dummy_list_data, device):
    """Tests audioDataset initialization with list data."""
    X_list_np, y_tensor = dummy_list_data
    dataset = audioDataset(X_list_np, y_tensor, device=device)
    assert len(dataset) == 5
    assert dataset.X is X_list_np  # Should store the original list
    assert torch.equal(dataset.y, y_tensor)  # y should be tensor on device
    assert dataset.device == device


def test_audioDataset_len_numpy(dummy_numpy_data, device):
    """Tests audioDataset __len__ with numpy data."""
    X_np, _ = dummy_numpy_data
    dataset = audioDataset(X_np, device=device)
    assert len(dataset) == X_np.shape[0]


def test_audioDataset_len_list(dummy_list_data, device):
    """Tests audioDataset __len__ with list data."""
    X_list_np, _ = dummy_list_data
    dataset = audioDataset(X_list_np, device=device)
    assert len(dataset) == len(X_list_np)


def test_audioDataset_getitem_numpy(dummy_numpy_data, device):
    """Tests audioDataset __getitem__ with numpy data."""
    X_np, y_tensor = dummy_numpy_data
    dataset = audioDataset(X_np, y_tensor, device=device)
    sample_x, sample_y = dataset[0]
    assert torch.is_tensor(sample_x)
    assert torch.is_tensor(sample_y)
    assert sample_x.device == device
    assert sample_y.device == device
    # Adjust expected shape based on fixture data
    assert sample_x.shape == (1, 1600)
    assert sample_y.shape == (3,)
    # Compare tensor content
    assert torch.equal(torch.from_numpy(X_np[0]).to(device), sample_x)
    assert torch.equal(y_tensor[0], sample_y)


def test_audioDataset_getitem_list(dummy_list_data, device):
    """Tests audioDataset __getitem__ with list data."""
    X_list_np, y_tensor = dummy_list_data
    dataset = audioDataset(X_list_np, y_tensor, device=device)
    sample_x, sample_y = dataset[0]
    assert torch.is_tensor(sample_x)
    assert torch.is_tensor(sample_y)
    assert sample_x.device == device
    assert sample_y.device == device
    assert sample_x.shape == (1, 1600)
    assert sample_y.shape == (3,)
    assert torch.equal(torch.from_numpy(X_list_np[0]).to(device), sample_x)
    assert torch.equal(y_tensor[0], sample_y)


def test_audioDataset_getitem_no_y(dummy_numpy_data, device):
    """Tests audioDataset __getitem__ when y is None."""
    X_np, _ = dummy_numpy_data
    dataset = audioDataset(X_np, device=device)  # y is None
    sample_x, sample_y = dataset[0]
    assert torch.is_tensor(sample_x)
    assert torch.is_tensor(sample_y)
    assert sample_x.device == device
    assert sample_y.device == device
    assert sample_x.shape == (1, 1600)
    # Default y is torch.zeros((2)) on the device
    assert torch.equal(sample_y, torch.zeros((2), device=device))


def test_audioDataset_transform(dummy_numpy_data, device):
    """Tests audioDataset with a transform."""
    X_np, y_tensor = dummy_numpy_data
    # Mock transform needs to return tensors or tensor-like objects for device check
    transformed_x_mock = torch.rand(1, 100,
                                    device=device)  # Dummy tensor output
    transformed_y_mock = torch.rand(3, device=device)
    mock_transform = MagicMock(return_value=(transformed_x_mock,
                                             transformed_y_mock))
    dataset = audioDataset(X_np,
                           y_tensor,
                           transform=mock_transform,
                           device=device)
    sample_x, sample_y = dataset[0]
    mock_transform.assert_called_once()
    # Check the input to the transform (should be tensors on the correct device)
    call_args = mock_transform.call_args[0][0]
    assert torch.is_tensor(call_args[0])
    assert torch.is_tensor(call_args[1])
    assert call_args[0].device == device
    assert call_args[1].device == device
    assert torch.equal(torch.from_numpy(X_np[0]).to(device), call_args[0])
    assert torch.equal(y_tensor[0], call_args[1])
    # Check the output
    assert sample_x is transformed_x_mock  # Check object identity for mock output
    assert sample_y is transformed_y_mock


def test_audioDataset_data_by_reference(mock_data_reference, device):
    """Tests audioDataset with data_by_reference=True."""
    X_mock, y_tensor, expected_np_data = mock_data_reference
    dataset = audioDataset(X_mock,
                           y_tensor,
                           data_by_reference=True,
                           mono=True,
                           device=device)
    assert len(dataset) == 5
    sample_x, sample_y = dataset[0]

    # Check that get_data_by_value was called on the mock object with np.float32
    X_mock[0].get_data_by_value.assert_called_once_with(dtype=np.float32,
                                                        mono=True)

    # Check the sample values
    expected_x_np = expected_np_data[0]  # Get the numpy data mock returned
    assert torch.is_tensor(sample_x)
    assert torch.is_tensor(sample_y)
    assert sample_x.device == device
    assert sample_y.device == device
    # Convert expected numpy array to tensor on device for comparison
    expected_x_tensor = torch.from_numpy(expected_x_np).to(device)
    assert torch.equal(expected_x_tensor, sample_x)
    assert torch.equal(y_tensor[0], sample_y)


# --- Tests for AugmentingAudioDataset ---


@pytest.fixture
def augmenting_dataset_numpy(dummy_numpy_data, device):
    """Creates an AugmentingAudioDataset with numpy data."""
    X_np, y_tensor = dummy_numpy_data
    return AugmentingAudioDataset(X_np,
                                  y_tensor,
                                  sampling_rate=16000,
                                  mono=True,
                                  device=device)


@pytest.fixture
def augmenting_dataset_list(dummy_list_data, device):
    """Creates an AugmentingAudioDataset with list data."""
    X_list_np, y_tensor = dummy_list_data
    return AugmentingAudioDataset(X_list_np,
                                  y_tensor,
                                  sampling_rate=16000,
                                  mono=True,
                                  device=device)


@pytest.fixture
def augmenting_dataset_ref(mock_data_reference, device):
    """Creates an AugmentingAudioDataset with mock reference data."""
    X_mock, y_tensor, _ = mock_data_reference
    return AugmentingAudioDataset(X_mock,
                                  y_tensor,
                                  data_by_reference=True,
                                  sampling_rate=16000,
                                  mono=True,
                                  device=device)


def test_augmentingAudioDataset_init_numpy(dummy_numpy_data, device):
    """Tests AugmentingAudioDataset initialization with numpy data."""
    X_np, y_tensor = dummy_numpy_data
    dataset = AugmentingAudioDataset(X_np,
                                     y_tensor,
                                     sampling_rate=16000,
                                     mono=True,
                                     device=device)
    assert len(dataset) == 5
    assert dataset.X is X_np
    assert torch.equal(dataset.y, y_tensor)
    assert dataset.sampling_rate == 16000
    assert dataset.transform is None
    assert dataset.batch_transforms == []
    assert not dataset.data_by_reference
    assert dataset.device == device


def test_augmentingAudioDataset_init_list(dummy_list_data, device):
    """Tests AugmentingAudioDataset initialization with list data."""
    X_list_np, y_tensor = dummy_list_data
    dataset = AugmentingAudioDataset(X_list_np,
                                     y_tensor,
                                     sampling_rate=16000,
                                     mono=True,
                                     device=device)
    assert len(dataset) == 5
    assert dataset.X is X_list_np
    assert torch.equal(dataset.y, y_tensor)
    assert dataset.sampling_rate == 16000
    assert dataset.device == device


def test_augmentingAudioDataset_init_batch_transforms(dummy_numpy_data, device):
    """Tests AugmentingAudioDataset initialization with batch transforms."""
    X_np, y_tensor = dummy_numpy_data
    transforms = ['random_mergev2']
    dataset = AugmentingAudioDataset(X_np,
                                     y_tensor,
                                     sampling_rate=16000,
                                     batch_transforms=transforms,
                                     mono=True,
                                     device=device)
    assert dataset.batch_transforms == transforms


def test_augmentingAudioDataset_len_numpy(augmenting_dataset_numpy):
    """Tests AugmentingAudioDataset __len__ with numpy data."""
    assert len(augmenting_dataset_numpy) == 5


def test_augmentingAudioDataset_len_list(augmenting_dataset_list):
    """Tests AugmentingAudioDataset __len__ with list data."""
    assert len(augmenting_dataset_list) == 5


def test_augmentingAudioDataset_get_x_numpy(augmenting_dataset_numpy):
    """Tests AugmentingAudioDataset get_x with numpy data."""
    dataset = augmenting_dataset_numpy
    x = dataset.get_x(0)
    assert torch.is_tensor(x)  # Should return tensor
    assert x.device == dataset.device
    assert x.shape == (1, 1600)
    # Compare tensor content
    assert torch.equal(x, torch.from_numpy(dataset.X[0]).to(dataset.device))


def test_augmentingAudioDataset_get_x_list(augmenting_dataset_list):
    """Tests AugmentingAudioDataset get_x with list data."""
    dataset = augmenting_dataset_list
    x = dataset.get_x(0)
    assert torch.is_tensor(x)  # Should return tensor
    assert x.device == dataset.device
    assert x.shape == (1, 1600)
    # Compare tensor content
    assert torch.equal(x, torch.from_numpy(dataset.X[0]).to(dataset.device))


def test_augmentingAudioDataset_get_x_ref(augmenting_dataset_ref,
                                          mock_data_reference, device):
    """Tests AugmentingAudioDataset get_x with data_by_reference."""
    dataset = augmenting_dataset_ref
    _, _, expected_np_data = mock_data_reference  # Get the expected numpy array
    x = dataset.get_x(0)

    # Check mock call expects np.float32
    dataset.X[0].get_data_by_value.assert_called_once_with(dtype=np.float32,
                                                           mono=True)

    expected_x_np = expected_np_data[0]  # Get specific numpy array
    assert torch.is_tensor(x)  # Should return tensor
    assert x.device == device
    assert x.shape == (1, 1600)  # Check shape consistency
    # Compare tensor content
    assert torch.equal(x, torch.from_numpy(expected_x_np).to(device))


def test_augmentingAudioDataset_get_sample_numpy(augmenting_dataset_numpy):
    """Tests AugmentingAudioDataset get_sample with numpy data."""
    dataset = augmenting_dataset_numpy
    x, y = dataset.get_sample(0)
    assert torch.is_tensor(x)  # Should return tensor
    assert torch.is_tensor(y)
    assert x.device == dataset.device
    assert y.device == dataset.device
    assert x.shape == (1, 1600)
    assert y.shape == (3,)
    # Compare tensor content
    assert torch.equal(x, torch.from_numpy(dataset.X[0]).to(dataset.device))
    assert torch.equal(y, dataset.y[0])


def test_augmentingAudioDataset_get_sample_list(augmenting_dataset_list):
    """Tests AugmentingAudioDataset get_sample with list data."""
    dataset = augmenting_dataset_list
    x, y = dataset.get_sample(0)
    assert torch.is_tensor(x)  # Should return tensor
    assert torch.is_tensor(y)
    assert x.device == dataset.device
    assert y.device == dataset.device
    assert x.shape == (1, 1600)
    assert y.shape == (3,)
    # Compare tensor content
    assert torch.equal(x, torch.from_numpy(dataset.X[0]).to(dataset.device))
    assert torch.equal(y, dataset.y[0])


def test_augmentingAudioDataset_get_sample_ref(augmenting_dataset_ref,
                                               mock_data_reference, device):
    """Tests AugmentingAudioDataset get_sample with data_by_reference."""
    dataset = augmenting_dataset_ref
    _, _, expected_np_data = mock_data_reference  # Get expected numpy
    x, y = dataset.get_sample(0)

    # Check mock call expects np.float32
    dataset.X[0].get_data_by_value.assert_called_once_with(dtype=np.float32,
                                                           mono=True)

    expected_x_np = expected_np_data[0]
    assert torch.is_tensor(x)  # Should return tensor
    assert torch.is_tensor(y)
    assert x.device == device
    assert y.device == device
    assert x.shape == (1, 1600)  # Check shape consistency
    # Compare tensor content
    assert torch.equal(x, torch.from_numpy(expected_x_np).to(device))
    assert torch.equal(y, dataset.y[0])


def test_augmentingAudioDataset_getitem_no_batch_transform(
        augmenting_dataset_numpy):
    """Tests __getitem__ when no batch transforms are applied."""
    dataset = augmenting_dataset_numpy
    device = dataset.device
    # Ensure no merge happens
    dataset.batch_transforms = []
    # Mock get_sample to ensure it's called (it now returns tensors)
    original_x_tensor, original_y_tensor = dataset.get_sample(0)
    with patch.object(dataset,
                      'get_sample',
                      return_value=(original_x_tensor,
                                    original_y_tensor)) as mock_get_sample:
        sample_x, sample_y = dataset[0]
        mock_get_sample.assert_called_once_with(0)
        assert torch.is_tensor(sample_x)
        assert torch.is_tensor(sample_y)
        assert sample_x.device == device
        assert sample_y.device == device
        assert sample_x.shape == (1, 1600)
        assert sample_y.shape == (3,)
        # Check output is the same as get_sample returned
        assert torch.equal(sample_x, original_x_tensor)
        assert torch.equal(sample_y, original_y_tensor)


def test_augmentingAudioDataset_getitem_with_transform(
        augmenting_dataset_numpy):
    """Tests __getitem__ with a standard torch transform."""
    dataset = augmenting_dataset_numpy
    device = dataset.device
    dataset.batch_transforms = []  # No batch transforms
    # Mock transform needs to return tensors on the correct device
    transformed_x_mock = torch.rand(1, 100, device=device)
    transformed_y_mock = torch.rand(3, device=device)
    mock_transform = MagicMock(return_value=(transformed_x_mock,
                                             transformed_y_mock))
    dataset.transform = mock_transform

    sample_x, sample_y = dataset[0]

    mock_transform.assert_called_once()
    call_args = mock_transform.call_args[0][0]
    # Input to transform should be tensors from get_sample
    original_x_tensor = torch.from_numpy(dataset.X[0]).to(device)
    original_y_tensor = dataset.y[0]
    assert torch.is_tensor(call_args[0])
    assert torch.is_tensor(call_args[1])
    assert call_args[0].device == device
    assert call_args[1].device == device
    assert torch.equal(original_x_tensor, call_args[0])
    assert torch.equal(original_y_tensor, call_args[1])
    # Output should be the transformed data
    assert sample_x is transformed_x_mock
    assert sample_y is transformed_y_mock


def test_augmentingAudioDataset_merge_samples(augmenting_dataset_numpy):
    """Tests the merge_samples method."""
    dataset = augmenting_dataset_numpy
    device = dataset.device
    idx1, idx2 = 0, 1
    # Get tensors directly using get_x/y
    x1 = dataset.get_x(idx1)
    x2 = dataset.get_x(idx2)
    y1 = dataset.y[idx1]
    y2 = dataset.y[idx2]

    # Mock torch.rand to control merge percentage
    percentage = 0.5
    with patch('torch.rand',
               return_value=torch.tensor([0.15 + percentage * (0.85 - 0.15)],
                                         device=device)):  # Inverse calculation
        merged_x, merged_y = dataset.merge_samples(idx1, idx2)

    # Calculate expected values using tensors
    expected_x = (x1 * (1 - percentage)) + (x2 * percentage)
    expected_y = torch.clamp((y1 + y2).float(), max=1.0).to(y1.dtype)

    assert torch.is_tensor(merged_x)  # Output should be tensor
    assert torch.is_tensor(merged_y)
    assert merged_x.device == device
    assert merged_y.device == device
    # Use torch.allclose for float comparison
    assert torch.allclose(merged_x, expected_x)
    assert torch.equal(merged_y, expected_y)


def test_augmentingAudioDataset_merge_samples_with_silence(
        augmenting_dataset_numpy):
    """Tests merge_samples with non-associative labels (silence)."""
    dataset = augmenting_dataset_numpy
    device = dataset.device
    # Assume class 0 is silence
    silence_index = 0
    dataset.non_associative_labels = torch.tensor([silence_index],
                                                  dtype=torch.long,
                                                  device=device)

    # Ensure y has silence correctly set up
    dataset.y = torch.tensor(
        [[1., 1., 0.], [1., 0., 1.], [0., 1., 1.], [0., 0., 0.], [0., 1., 0.]],
        device=device)

    # Case 1: Merge non-silence (idx=2) with silence (idx=0)
    idx1, idx2 = 2, 0  # idx2 (sample 0) is silence
    x1 = dataset.get_x(idx1)
    x2 = dataset.get_x(idx2)
    y1 = dataset.y[idx1]  # [0., 1., 1.]
    y2 = dataset.y[idx2]  # [1., 1., 0.] (silence + class 1)

    # Mock torch.rand for merge percentage and flip_coin
    merge_percentage = 0.5
    # Mock for flip_coin: > 0.5 -> keep merged y, don't force silence
    flip_coin_rand_val = torch.tensor([0.7], device=device)
    # Mock for merge percentage calculation
    merge_rand_val = torch.tensor([0.15 + merge_percentage * (0.85 - 0.15)],
                                  device=device)

    with patch('torch.rand', side_effect=[merge_rand_val, flip_coin_rand_val]):
        # Remove non_associative_labels from call
        merged_x, merged_y = dataset.merge_samples(idx1, idx2)

    # Calculate expected results with tensors
    expected_x = (x1 * (1 - merge_percentage)) + (x2 * merge_percentage)
    # Calculate intermediate merged y
    y_merged_intermediate = torch.clamp((y1 + y2).float(),
                                        max=1.0).to(y1.dtype)  # [1., 1., 1.]
    # Expected after flip_coin (force non-silence)
    expected_y = y_merged_intermediate.clone()
    expected_y[silence_index] = 0  # [0., 1., 1.]

    assert torch.allclose(merged_x, expected_x, atol=1e-6)
    assert torch.equal(merged_y, expected_y)

    # Case 2: Force silence output
    # Mock for flip_coin: <= 0.5 -> force silence
    flip_coin_rand_val = torch.tensor([0.3], device=device)
    with patch('torch.rand', side_effect=[merge_rand_val, flip_coin_rand_val]):
        # Remove non_associative_labels from call
        merged_x, merged_y = dataset.merge_samples(idx1, idx2)

    # If flip_coin forces silence, and only one input was silent (y2), use x2
    expected_x = x2
    expected_y = torch.zeros_like(y1)
    expected_y[silence_index] = 1  # Forced silence

    assert torch.allclose(merged_x, expected_x, atol=1e-6)
    assert torch.equal(merged_y, expected_y)

    # Case 3: Both are silence
    idx1, idx2 = 0, 1  # Samples 0 and 1 are both silence (and have other labels)
    dataset.y[1] = torch.tensor([1., 0., 0.],
                                device=device)  # Make sample 1 only silence
    x1 = dataset.get_x(idx1)  # Sample 0: silence + class 1
    x2 = dataset.get_x(idx2)  # Sample 1: silence only
    y1 = dataset.y[idx1]  # [1., 1., 0.]
    y2 = dataset.y[idx2]  # [1., 0., 0.]

    # Mock for flip_coin: <= 0.5 -> force silence
    flip_coin_rand_val = torch.tensor([0.3], device=device)
    with patch('torch.rand', side_effect=[merge_rand_val, flip_coin_rand_val]):
        # Remove non_associative_labels from call
        merged_x, merged_y = dataset.merge_samples(idx1, idx2)

    # If both were silence, merge happens normally, flip_coin outcome determines y
    expected_x = (x1 * (1 - merge_percentage)) + (x2 * merge_percentage)
    expected_y = torch.zeros_like(y1)
    expected_y[silence_index] = 1  # Expect silence because flip coin forced it

    assert torch.allclose(merged_x, expected_x, atol=1e-6)
    assert torch.equal(merged_y, expected_y)


def test_augmentingAudioDataset_random_merge(augmenting_dataset_numpy):
    """Tests the random_merge method."""
    dataset = augmenting_dataset_numpy
    device = dataset.device
    # Setup non_associative_labels directly on the instance
    dataset.non_associative_labels = torch.tensor([0],
                                                  dtype=torch.long,
                                                  device=device)

    # Mock random.randint and merge_samples
    mock_merged_x = torch.rand(1, 1600, device=device)
    mock_merged_y = torch.rand(3, device=device)
    with patch('random.randint', side_effect=[0, 1]) as mock_randint, \
         patch('torch.rand', return_value=torch.tensor([0.3])) as mock_torch_rand, \
         patch.object(dataset, 'merge_samples', return_value=(mock_merged_x, mock_merged_y)) as mock_merge:
        x, y = dataset.random_merge(
            merge_probability=0.5
        )  # merge_probability = 0.5, rand=0.3 -> should merge

        mock_torch_rand.assert_called_once()  # Check torch.rand was called
        mock_randint.assert_any_call(0, len(dataset) - 1)
        assert mock_randint.call_count == 2
        # Check merge_samples call signature (no non_associative_labels)
        mock_merge.assert_called_once_with(0, 1)
        assert x is mock_merged_x
        assert y is mock_merged_y


def test_augmentingAudioDataset_random_mergev2_no_merge(
        augmenting_dataset_numpy):
    """Tests random_mergev2 when merge probability is low."""
    dataset = augmenting_dataset_numpy
    device = dataset.device
    original_len = len(dataset)
    # Use a real list for id_mix
    dataset.id_mix = list(range(original_len))
    pop_return_value = dataset.id_mix[-1]  # Value that will be popped

    # Mock torch.rand to prevent merge
    with patch('torch.rand', return_value=torch.tensor([0.8], device=device)) as mock_torch_rand, \
         patch.object(dataset, 'get_sample') as mock_get_sample, \
         patch.object(dataset, 'merge_samples') as mock_merge:

        # Call the method
        dataset.random_mergev2(
            merge_probability=0.5)  # rand=0.8 > prob=0.5 -> no merge

        mock_torch_rand.assert_called_once()
        # Check get_sample was called with the popped ID
        mock_get_sample.assert_called_once_with(pop_return_value)
        mock_merge.assert_not_called()
        assert len(dataset.id_mix) == original_len - 1  # Check list length


def test_augmentingAudioDataset_random_mergev2_with_merge(
        augmenting_dataset_numpy):
    """Tests random_mergev2 when merge probability is high."""
    dataset = augmenting_dataset_numpy
    device = dataset.device
    original_len = len(dataset)
    dataset.id_mix = list(range(original_len))
    pop_id1 = dataset.id_mix[-1]
    pop_id2 = dataset.id_mix[-2]

    # Mock torch.rand to force merge
    mock_merged_x = torch.rand(1, 1600, device=device)
    mock_merged_y = torch.rand(3, device=device)
    with patch('torch.rand', return_value=torch.tensor([0.3], device=device)) as mock_torch_rand, \
         patch.object(dataset, 'merge_samples', return_value=(mock_merged_x, mock_merged_y)) as mock_merge:

        x, y = dataset.random_mergev2(
            merge_probability=0.5)  # rand=0.3 < prob=0.5 -> merge

        mock_torch_rand.assert_called_once()
        # Check merge_samples was called with the correct popped IDs (in reverse order of popping)
        mock_merge.assert_called_once_with(pop_id1, pop_id2)
        assert x is mock_merged_x
        assert y is mock_merged_y
        assert len(dataset.id_mix) == original_len - 2  # Check list length


def test_augmentingAudioDataset_random_mergev2_shuffle(
        augmenting_dataset_numpy):
    """Tests random_mergev2 triggers shuffle when id_mix is empty/small."""
    dataset = augmenting_dataset_numpy
    device = dataset.device
    original_len = len(dataset)

    # --- Test case 1: Shuffle because id_mix is empty initially ---
    dataset.id_mix = []
    # Mock torch.rand to force merge
    with patch('torch.rand', return_value=torch.tensor([0.3], device=device)) as mock_torch_rand, \
         patch.object(dataset, 'shuffle_indexes', wraps=dataset.shuffle_indexes) as mock_shuffle, \
         patch.object(dataset, 'merge_samples', return_value=('merged_x', 'merged_y')) as mock_merge:

        assert len(dataset.id_mix) == 0
        dataset.random_mergev2()  # Should trigger shuffle

        mock_shuffle.assert_called_once()
        assert len(dataset.id_mix
                  ) == original_len - 2  # Should have popped 2 after shuffle
        mock_merge.assert_called_once()  # Merge should have happened

    # --- Test case 2: Shuffle because id_mix has only 1 element before merge ---
    dataset.id_mix = [0]  # Only one element left
    with patch('torch.rand', return_value=torch.tensor([0.3], device=device)) as mock_torch_rand, \
         patch.object(dataset, 'shuffle_indexes', wraps=dataset.shuffle_indexes) as mock_shuffle, \
         patch.object(dataset, 'merge_samples', return_value=('merged_x', 'merged_y')) as mock_merge:

        assert len(dataset.id_mix) == 1
        dataset.random_mergev2(
        )  # Should trigger shuffle when needing the second ID

        mock_shuffle.assert_called_once()
        # Should have popped 1 (original 0), shuffled, then popped 1 more from shuffled list
        assert len(
            dataset.id_mix
        ) == original_len - 2  # Corrected assertion: 2 items are popped after shuffle
        mock_merge.assert_called_once()  # Merge should have happened


def test_augmentingAudioDataset_getitem_with_noise(augmenting_dataset_numpy):
    """Tests __getitem__ using the AddGaussianNoise augmentation."""
    dataset = augmenting_dataset_numpy
    device = dataset.device
    dataset.batch_transforms = ['AddGaussianNoise']

    # Get original sample tensors
    original_x_tensor, original_y_tensor = dataset.get_sample(0)

    # Mock the internal noise function
    noisy_tensor = torch.rand(1, 1600, dtype=torch.float32,
                              device=device)  # Dummy noisy tensor
    with patch.object(dataset, '_apply_gaussian_noise', return_value=noisy_tensor) as mock_noise_applier, \
         patch.object(dataset, 'get_merged_sample', return_value=None) as mock_get_merged: # Ensure get_sample is called

        sample_x, sample_y = dataset[0]

        mock_get_merged.assert_called_once()  # Ensure merge path wasn't taken
        # Assert the internal method was called with the correct tensor input
        mock_noise_applier.assert_called_once()
        call_args, _ = mock_noise_applier.call_args
        assert torch.is_tensor(call_args[0])
        assert torch.equal(call_args[0], original_x_tensor)

        # Check the output is the tensor returned by the mock
        assert torch.equal(sample_x, noisy_tensor)
        assert torch.equal(sample_y, original_y_tensor)  # y should be unchanged


def test_augmentingAudioDataset_getitem_with_merge_and_noise(
        augmenting_dataset_numpy):
    """Tests __getitem__ with both merging and noise."""
    dataset = augmenting_dataset_numpy
    device = dataset.device
    dataset.batch_transforms = ['random_mergev2', 'AddGaussianNoise']

    # Mock the internal noise function
    noisy_merged_tensor = torch.rand(1,
                                     1600,
                                     dtype=torch.float32,
                                     device=device)  # Dummy noisy tensor
    mock_noise_applier = MagicMock(return_value=noisy_merged_tensor)
    dataset._apply_gaussian_noise = mock_noise_applier

    # Mock the merge function to return tensors
    merged_x_tensor = torch.rand(1, 1600, dtype=torch.float32, device=device)
    merged_y_tensor = torch.tensor([0., 1., 0.], device=device)
    with patch.object(dataset,
                      'random_mergev2',
                      return_value=(merged_x_tensor,
                                    merged_y_tensor)) as mock_merge:

        sample_x, sample_y = dataset[0]

        mock_merge.assert_called_once()  # Ensure merge was called

        # Assert the internal noise method was called with the merged tensor
        mock_noise_applier.assert_called_once()
        call_args, _ = mock_noise_applier.call_args
        assert torch.is_tensor(call_args[0])
        assert torch.equal(call_args[0], merged_x_tensor)

        # Check the output is the tensor returned by the noise mock
        assert torch.equal(sample_x, noisy_merged_tensor)
        assert torch.equal(sample_y, merged_y_tensor)  # y comes from merge step


def test_flip_coin_for_silence(device):
    """Tests the flip_coin_for_silence helper."""
    # Create a dummy dataset instance just to call the method
    dataset = AugmentingAudioDataset(
        X=[1], y=[1], mono=True,
        device=device)  # Minimal init, ensure mono=True
    silence_index = 0

    # Case 1: Input is silence, coin forces non-silence (rand > prob)
    y_in = torch.tensor([1.0, 0.0, 1.0], device=device)
    # Mock torch.rand to return > 0.5
    with patch('torch.rand', return_value=torch.tensor([0.7], device=device)):
        y_out = dataset.flip_coin_for_silence(y_in.clone(),
                                              silence_index,
                                              prob=0.5)
    expected_y = torch.tensor([0.0, 0.0, 1.0], device=device)
    assert torch.equal(y_out, expected_y)

    # Case 2: Input is silence, coin forces silence (rand <= prob)
    y_in = torch.tensor([1.0, 0.0, 1.0], device=device)
    # Mock torch.rand to return <= 0.5
    with patch('torch.rand', return_value=torch.tensor([0.3], device=device)):
        y_out = dataset.flip_coin_for_silence(y_in.clone(),
                                              silence_index,
                                              prob=0.5)
    expected_y = torch.tensor([1.0, 0.0, 0.0],
                              device=device)  # Only silence remains
    assert torch.equal(y_out, expected_y)

    # Case 3: Input is not silence
    y_in = torch.tensor([0.0, 1.0, 1.0], device=device)
    with patch('torch.rand') as mock_rand:  # Shouldn't be called
        y_out = dataset.flip_coin_for_silence(y_in.clone(),
                                              silence_index,
                                              prob=0.5)
    mock_rand.assert_not_called()
    assert torch.equal(y_out, y_in)  # Output unchanged


def test_augmentingAudioDataset_getitem_with_mix_channels(device):
    """Tests __getitem__ with mix_channels augmentation."""
    # Need 2 channels for mix_channels
    X_2ch_np = np.random.rand(5, 2, 1600).astype(np.float32)
    y_np = np.random.randint(0, 2, (5, 3)).astype(np.float32)
    # Convert initial data to tensors on device for dataset init
    y_tensor = torch.from_numpy(y_np).to(device)
    dataset = AugmentingAudioDataset(
        X_2ch_np,  # Keep X as numpy to test loading
        y_tensor,
        sampling_rate=16000,
        mono=False,
        batch_transforms=['mix_channels'],
        mix_channels_coeff=0.3,
        device=device)

    # We need the original sample TENSORS to compare against the mix_channels input
    # Use get_sample directly for this, before calling __getitem__
    original_x_tensor, original_y_tensor = dataset.get_sample(0)
    assert original_x_tensor.shape[0] == 2  # Ensure get_sample loaded 2 channels

    # Mock random.randint which is used inside mix_channels
    # Also mock get_merged_sample to ensure get_sample path is taken
    # Mock torch.rand for the decision within mix_channels
    with patch.object(dataset, 'get_merged_sample', return_value=None) as mock_get_merged, \
         patch('torch.rand', return_value=torch.tensor([0.2], device=device)): # Force first mix formula (<0.5)

        sample_x, sample_y = dataset[0]  # Call __getitem__

        mock_get_merged.assert_called_once()  # Ensure merge wasn't used

        # Check output
        assert torch.is_tensor(sample_x)
        assert sample_x.device == device
        # Shape should be (1, length) after mixing
        assert sample_x.shape == (1, 1600)
        assert torch.equal(sample_y,
                           original_y_tensor)  # y is unchanged by mix_channels

        # Verify the mixing calculation using tensors
        # Use the first mix formula because torch.rand was mocked < 0.5
        coeff = dataset.mix_channels_coeff
        # Use the tensor input
        expected_mixed_x_tensor = original_x_tensor[0, :] * (
            1 - coeff) + original_x_tensor[1, :] * coeff
        # Add channel dim back
        expected_mixed_x_tensor = expected_mixed_x_tensor.unsqueeze(0)
        # Use torch.allclose for floating point comparison
        assert torch.allclose(sample_x, expected_mixed_x_tensor, atol=1e-6)


# --- New Tests for Enhanced Coverage ---


@pytest.mark.parametrize("test_dtype", [torch.float32, torch.float16])
def test_audioDataset_different_dtype(dummy_numpy_data, device, test_dtype):
    """Tests audioDataset with different xdtypes."""
    # Skip float16 on CPU if not supported well or for simplicity if desired
    if test_dtype == torch.float16 and device == torch.device('cpu'):
        pytest.skip(
            "float16 tests on CPU might have precision issues or lack full support"
        )

    X_np, y_tensor = dummy_numpy_data
    # Ensure y_tensor matches device
    y_tensor = y_tensor.to(device)

    dataset = audioDataset(X_np, y_tensor, xdtype=test_dtype, device=device)
    sample_x, sample_y = dataset[0]

    assert torch.is_tensor(sample_x)
    assert torch.is_tensor(sample_y)
    assert sample_x.device == device
    assert sample_y.device == device
    assert sample_x.dtype == test_dtype
    assert sample_y.dtype == y_tensor.dtype  # y dtype should remain unchanged

    # Check content conversion
    expected_x_tensor = torch.from_numpy(X_np[0]).to(device=device,
                                                     dtype=test_dtype)
    assert torch.allclose(sample_x, expected_x_tensor,
                          atol=1e-3)  # Use allclose for float comparison
    assert torch.equal(sample_y, y_tensor[0])


@pytest.mark.parametrize("test_dtype", [torch.float32, torch.float16])
def test_augmentingAudioDataset_different_dtype(dummy_numpy_data, device,
                                                test_dtype):
    """Tests AugmentingAudioDataset basic get_sample with different xdtypes."""
    if test_dtype == torch.float16 and device == torch.device('cpu'):
        pytest.skip(
            "float16 tests on CPU might have precision issues or lack full support"
        )

    X_np, y_tensor = dummy_numpy_data
    y_tensor = y_tensor.to(device)

    dataset = AugmentingAudioDataset(X_np,
                                     y_tensor,
                                     xdtype=test_dtype,
                                     mono=True,
                                     device=device,
                                     sampling_rate=16000)
    sample_x, sample_y = dataset.get_sample(0)  # Test get_sample directly

    assert torch.is_tensor(sample_x)
    assert torch.is_tensor(sample_y)
    assert sample_x.device == device
    assert sample_y.device == device
    assert sample_x.dtype == test_dtype
    assert sample_y.dtype == y_tensor.dtype

    expected_x_tensor = torch.from_numpy(X_np[0]).to(device=device,
                                                     dtype=test_dtype)
    assert torch.allclose(sample_x, expected_x_tensor, atol=1e-3)
    assert torch.equal(sample_y, y_tensor[0])


# --- Dummy RecordingsDataset for Testing ---
class DummyRecordingsDataset(Dataset):

    def __init__(self,
                 num_samples=5,
                 channels=1,
                 length=1600,
                 sample_rate=16000,
                 return_numpy=True,
                 device=torch.device('cpu')):
        self.num_samples = num_samples
        self.channels = channels
        self.length = length
        self.sample_rate = sample_rate
        self.return_numpy = return_numpy
        self.device = device
        # Generate dummy data (can be numpy or tensor based on return_numpy)
        self._data = []
        self._labels = []
        for i in range(num_samples):
            x_np = np.random.rand(channels, length).astype(np.float32)
            y_np = np.random.randint(0, 2, (3,)).astype(np.float32)
            if return_numpy:
                self._data.append(x_np)
                self._labels.append(y_np)
            else:
                self._data.append(torch.from_numpy(x_np).to(device))
                self._labels.append(torch.from_numpy(y_np).to(device))
        # Ensure some silence label for relevant tests
        if return_numpy:
            self._labels[0][0] = 1.0
            self._labels[1][0] = 1.0
        else:
            self._labels[0][0] = 1.0
            self._labels[1][0] = 1.0

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Mimic returning data and label
        x = self._data[idx]
        y = self._labels[idx]
        # Optionally mimic RecordingsDataset returning None for label
        # if idx % 2 == 0: # Example condition
        #    y = None
        return x, y


# ------------------------------------------


@pytest.fixture
def dummy_recordings_dataset_numpy_input(device):
    """Fixture providing a DummyRecordingsDataset that returns numpy arrays."""
    return DummyRecordingsDataset(return_numpy=True, device=device)


@pytest.fixture
def dummy_recordings_dataset_tensor_input(device):
    """Fixture providing a DummyRecordingsDataset that returns tensors."""
    return DummyRecordingsDataset(return_numpy=False, device=device)


def test_augmentingAudioDataset_get_x_recordings_numpy(
        dummy_recordings_dataset_numpy_input, device):
    """Tests get_x when X is a RecordingsDataset-like object returning numpy."""
    recordings_dataset = dummy_recordings_dataset_numpy_input
    dataset = AugmentingAudioDataset(
        recordings_dataset,
        y=None,  # y is ignored when X is RecordingsDataset
        sampling_rate=16000,
        mono=True,
        device=device)
    sample_x = dataset.get_x(0)

    assert torch.is_tensor(sample_x)
    assert sample_x.device == device
    assert sample_x.dtype == dataset.xdtype
    assert sample_x.shape == (recordings_dataset.channels,
                              recordings_dataset.length)
    # Check content
    expected_x_tensor = torch.from_numpy(recordings_dataset._data[0]).to(
        device=device, dtype=dataset.xdtype)
    assert torch.allclose(sample_x, expected_x_tensor)


def test_augmentingAudioDataset_get_sample_recordings_numpy(
        dummy_recordings_dataset_numpy_input, device):
    """Tests get_sample when X is a RecordingsDataset-like object returning numpy."""
    recordings_dataset = dummy_recordings_dataset_numpy_input
    dataset = AugmentingAudioDataset(recordings_dataset,
                                     y=None,
                                     sampling_rate=16000,
                                     mono=True,
                                     device=device)
    sample_x, sample_y = dataset.get_sample(0)

    assert torch.is_tensor(sample_x)
    assert torch.is_tensor(sample_y)
    assert sample_x.device == device
    assert sample_y.device == device
    assert sample_x.dtype == dataset.xdtype
    assert sample_y.dtype == torch.float32  # Labels usually float for BCELoss etc.
    assert sample_x.shape == (recordings_dataset.channels,
                              recordings_dataset.length)
    assert sample_y.shape == (3,)
    # Check content
    expected_x_tensor = torch.from_numpy(recordings_dataset._data[0]).to(
        device=device, dtype=dataset.xdtype)
    expected_y_tensor = torch.from_numpy(
        recordings_dataset._labels[0]).to(device=device)
    assert torch.allclose(sample_x, expected_x_tensor)
    assert torch.equal(sample_y, expected_y_tensor)


def test_augmentingAudioDataset_get_sample_recordings_tensor(
        dummy_recordings_dataset_tensor_input, device):
    """Tests get_sample when X is a RecordingsDataset-like object returning tensors."""
    recordings_dataset = dummy_recordings_dataset_tensor_input
    # Ensure the dummy dataset's device matches the test device
    assert recordings_dataset.device == device

    dataset = AugmentingAudioDataset(recordings_dataset,
                                     y=None,
                                     sampling_rate=16000,
                                     mono=True,
                                     device=device)
    sample_x, sample_y = dataset.get_sample(0)

    assert torch.is_tensor(sample_x)
    assert torch.is_tensor(sample_y)
    assert sample_x.device == device
    assert sample_y.device == device
    assert sample_x.dtype == dataset.xdtype  # Should match dataset's preference
    assert sample_y.dtype == recordings_dataset._labels[
        0].dtype  # Should keep original label dtype
    assert sample_x.shape == (recordings_dataset.channels,
                              recordings_dataset.length)
    assert sample_y.shape == (3,)
    # Check content (should be the tensors directly returned by the dummy dataset)
    # Need to ensure dtype matches for x
    assert torch.allclose(sample_x,
                          recordings_dataset._data[0].to(dtype=dataset.xdtype))
    assert torch.equal(sample_y, recordings_dataset._labels[0])


def test_augmentingAudioDataset_get_x_sampling_rate_warning(
        mock_data_reference, device, caplog):
    """Tests that get_x issues a warning for sampling rate mismatch with data_by_reference."""
    X_mock, y_tensor, _ = mock_data_reference
    target_sr = 16000
    actual_sr = 8000
    dataset = AugmentingAudioDataset(X_mock,
                                     y_tensor,
                                     data_by_reference=True,
                                     mono=True,
                                     sampling_rate=target_sr,
                                     device=device)

    # Modify the mock to return a different SR
    # Call get_data_by_value with mono=True to fetch the data for the mock setup
    original_data, _ = X_mock[0].get_data_by_value(mono=True)
    # Set the return value for the mock
    X_mock[0].get_data_by_value.return_value = (original_data, actual_sr)

    # Trigger the call within get_x
    _ = dataset.get_x(0)  # Call the function

    # Check if the warning was logged
    assert f'Warning: sampling_rate mismatch! Expected {target_sr}, got {actual_sr}' in caplog.text
