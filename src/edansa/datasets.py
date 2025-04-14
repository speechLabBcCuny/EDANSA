import random
from pathlib import Path
import logging  # Import logging

import numpy as np  # Keep numpy for initial loading if data_by_reference is used with external sources returning numpy

import torch
from torch.utils.data import Dataset

from edansa.dataimport import RecordingsDataset, RecordingsDatasetSSL


class audioDataset(
        Dataset
):  # Basic dataset - less critical for this refactor, but could be updated similarly

    def __init__(self,
                 X,
                 y=None,
                 transform=None,
                 data_by_reference=False,
                 non_associative_labels=None,
                 xdtype=torch.float32,
                 mono: bool = True,
                 device=torch.device('cpu')):  # Add device parameter
        '''
    Args:

    '''
        self.xdtype = xdtype
        self.X = X  # Keep X as is for now (could be list of paths or numpy array initially)
        # Convert y to tensor and move to device immediately if it exists
        if y is not None:
            # Ensure y is a tensor before moving
            if not isinstance(y, torch.Tensor):
                # Assuming y is a list/array of labels that can be converted
                # Adjust this conversion based on the actual structure of y
                try:
                    self.y = torch.tensor(y, device=device)  # Convert and move
                except TypeError as e:
                    # If y contains complex objects, this might need adjustment
                    print(
                        f"Warning: Could not automatically convert y to tensor. Ensure y elements are tensors or suitable for conversion. Error: {e}"
                    )
                    self.y = y  # Keep as is if conversion fails, handle downstream
            else:
                self.y = y.to(device)  # Move existing tensor
        else:
            self.y = None

        self.transform = transform
        self.data_by_reference = data_by_reference
        self.mono = mono
        if non_associative_labels is None:
            self.non_associative_labels = []
        else:
            # Assuming labels are indices, convert to tensor
            self.non_associative_labels = torch.tensor(non_associative_labels,
                                                       dtype=torch.long,
                                                       device=device)
        self.device = device  # Store the device

    def __len__(self):
        if isinstance(self.X, np.ndarray) or isinstance(self.X, torch.Tensor):
            return self.X.shape[0]
        else:  # Assuming list-like structure for paths or other refs
            return len(self.X)

    def get_x(self, idx):
        # Returns a tensor on self.device
        if self.data_by_reference:
            # Assuming self.X[idx].get_data_by_value returns numpy array and sr
            # This part might need adjustment based on the actual implementation of RecordingsDataset
            x_np, _ = self.X[idx].get_data_by_value(
                dtype=np.float32,  # Load as float32 numpy first
                mono=self.mono)
            x = torch.from_numpy(x_np).to(
                dtype=self.xdtype, device=self.device)  # Convert and move
        elif isinstance(self.X, torch.Tensor):
            x = self.X[idx].to(dtype=self.xdtype,
                               device=self.device)  # Ensure correct type/device
        elif isinstance(self.X, np.ndarray):
            # Convert numpy slice to tensor and move
            x = torch.from_numpy(self.X[idx]).to(dtype=self.xdtype,
                                                 device=self.device)
        else:
            # Assuming self.X contains tensors already or can be converted
            # This might need adjustment if self.X holds other types
            try:
                x = torch.tensor(self.X[idx],
                                 dtype=self.xdtype,
                                 device=self.device)
            except TypeError:
                # Fallback if direct conversion fails - assumes X[idx] is already a tensor
                x = self.X[idx].to(dtype=self.xdtype, device=self.device)

        # Ensure tensor has channel dimension if needed (e.g., shape [C, L])
        if x.ndim == 1:
            x = x.unsqueeze(0)  # Add channel dim: [L] -> [1, L]
        # elif x.ndim == 2 and self.channels > 0 and x.shape[0] != self.channels:
        #     # Handle potential shape mismatch (e.g. if loaded mono but channels=2 requested)
        #     # This might need specific logic based on how channel mismatches should be handled
        #     print(
        #         f"Warning: Loaded data channel ({x.shape[0]}) mismatch with requested channels ({self.channels}). Using loaded channels."
        #     )

        return x  # Tensor on self.device

    def __getitem__(self, idx):
        x_tensor = self.get_x(idx)  # Now returns tensor on self.device

        if self.y is None:
            # Create a dummy tensor on the correct device
            y_tensor = torch.zeros((2),
                                   device=self.device)  # Adjust size if needed
        else:
            y_tensor = self.y[idx]  # Already on self.device

        sample = x_tensor, y_tensor  # Sample contains tensors on self.device

        if self.transform:
            # Transform is applied to the tensor sample on self.device
            # Ensure transforms handle tensors on the correct device
            sample = self.transform(sample)

        return sample


class AugmentingAudioDataset(Dataset):

    def __init__(self,
                 X,
                 y=None,
                 transform=None,
                 batch_transforms=None,
                 sampling_rate=None,
                 mix_channels_coeff=None,
                 gauss_max_amplitude=0.015,
                 data_by_reference=False,
                 non_associative_labels=None,
                 xdtype=torch.float32,
                 mono=False,
                 device=torch.device('cpu')):  # Add device parameter
        '''
    Args:

    '''
        self.xdtype = xdtype
        self.device = device  # Store the device
        self.X = X  # Keep X as is for now (list of paths, numpy array, or potentially tensors)
        self.mono = mono
        try:
            # If mono is True, force 1 channel. Otherwise, try to get from shape dim 1.
            # Assumes X is array/tensor like (N, C, L) or (N, L) if not mono
            self.channels = 1 if mono else X.shape[1]
        except (AttributeError, IndexError):
            # If X has no shape (list, other dataset) or shape has no channel dim (e.g., N, L), default to 1 channel.
            # get_x will handle the actual tensor shape later.
            self.channels = 1
        # Convert y to tensor and move to device immediately if it exists
        if y is not None:
            # Ensure y is a tensor before moving
            if not isinstance(y, torch.Tensor):
                y = torch.tensor(y, dtype=torch.long, device=self.device)
            else:
                # If already a tensor, ensure it's on the correct device
                y = y.to(device=self.device)
        self.y = y

        self.transform = transform
        self.batch_transforms = batch_transforms if batch_transforms is not None else []
        self.sampling_rate = sampling_rate
        self.mix_channels_coeff = mix_channels_coeff
        self.gauss_max_amplitude = gauss_max_amplitude
        self.gauss_min_amplitude = 0.01
        self.gauss_prob = 0.5

        self.data_by_reference = data_by_reference
        self.id_mix = []

        if non_associative_labels is None:
            self.non_associative_labels = torch.tensor([],
                                                       dtype=torch.long,
                                                       device=self.device)
        else:
            # Ensure labels are tensor of indices on the correct device
            if isinstance(non_associative_labels, torch.Tensor):
                self.non_associative_labels = non_associative_labels.to(
                    dtype=torch.long, device=self.device)
            else:
                self.non_associative_labels = torch.tensor(
                    non_associative_labels,
                    dtype=torch.long,
                    device=self.device)

    def __len__(self):
        if isinstance(self.X, np.ndarray) or isinstance(self.X, torch.Tensor):
            return self.X.shape[0]
        else:  # Assuming list-like structure
            return len(self.X)

    def get_x(self, idx):
        # Returns a tensor on self.device
        if self.data_by_reference:
            # Assuming self.X[idx].get_data_by_value returns numpy array and sr
            # This part needs careful checking based on RecordingsDataset behavior
            x_np, sr = self.X[idx].get_data_by_value(
                dtype=np.float32,  # Load as float32 numpy
                mono=self.mono)
            if sr != self.sampling_rate:
                # Use logging instead of print
                logging.warning(
                    f'Warning: sampling_rate mismatch! Expected {self.sampling_rate}, got {sr} for index {idx}'
                )
            x = torch.from_numpy(x_np).to(
                dtype=self.xdtype, device=self.device)  # Convert and move
        # Check for Dataset instance *before* Tensor/ndarray/list checks
        elif isinstance(self.X, Dataset) and not isinstance(
                self.X, (audioDataset, AugmentingAudioDataset)):
            # Assume Dataset __getitem__ returns (data, label) or similar
            # We only need the data part here.
            x_data, _ = self.X[idx]
            if not isinstance(x_data, torch.Tensor):
                # If it returns numpy, convert it
                print(
                    f"Warning: Dataset at index {idx} did not return a tensor for x. Converting..."
                )
                x = torch.from_numpy(x_data).to(dtype=self.xdtype,
                                                device=self.device)
            else:
                # Ensure it's on the correct device and type
                x = x_data.to(dtype=self.xdtype, device=self.device)

        elif isinstance(self.X, torch.Tensor):
            # If X is already a tensor (potentially on CPU), get slice and move
            x = self.X[idx].to(dtype=self.xdtype, device=self.device)
        elif isinstance(self.X, np.ndarray):
            # Convert numpy slice to tensor and move
            x = torch.from_numpy(self.X[idx]).to(dtype=self.xdtype,
                                                 device=self.device)
        else:
            # Fallback: Assuming X holds list of tensors or things convertible to tensors
            try:
                # Try direct conversion if not already tensor
                if isinstance(self.X[idx], torch.Tensor):
                    x = self.X[idx].to(dtype=self.xdtype, device=self.device)
                else:
                    # This path assumes self.X[idx] is directly convertible (e.g., a list/tuple of numbers)
                    x = torch.tensor(self.X[idx],
                                     dtype=self.xdtype,
                                     device=self.device)
            except Exception as e:
                raise TypeError(
                    f"Unsupported data type in self.X at index {idx}: {type(self.X[idx])}. Error: {e}"
                )

        # Ensure tensor has channel dimension if needed (e.g., shape [C, L])
        if x.ndim == 1 and self.channels > 0:
            x = x.unsqueeze(0)  # Add channel dim: [L] -> [1, L]
        elif x.ndim == 2 and self.channels > 0:
            # Ensure correct number of channels, potentially repeating mono if needed
            if x.shape[0] == 1 and self.channels > 1:
                x = x.repeat(self.channels, 1)  # Repeat mono channel
            elif x.shape[0] != self.channels:
                print(
                    f"Warning: Loaded data channel ({x.shape[0]}) mismatch with requested channels ({self.channels}). Using loaded channels."
                )
                # Or potentially slice if x.shape[0] > self.channels: x = x[:self.channels, :]

        return x  # Tensor on self.device

    def get_sample(self, idx):
        # Returns (tensor, tensor) on self.device
        # Generalize this check for any Dataset instance, excluding self-types
        if isinstance(self.X, Dataset) and not isinstance(
                self.X, (audioDataset, AugmentingAudioDataset)):
            # Assume Dataset-like __getitem__ returns (data, label)
            # We need to ensure both are tensors on the correct device
            x_data, y_data = self.X[idx]

            # Ensure x is a tensor on the correct device/dtype
            if not isinstance(x_data, torch.Tensor):
                x_tensor = torch.from_numpy(x_data).to(dtype=self.xdtype,
                                                       device=self.device)
            else:
                x_tensor = x_data.to(dtype=self.xdtype, device=self.device)

            # Ensure y is a tensor on the correct device
            if y_data is None:  # Handle cases with no label (SSL)
                # Create a dummy tensor or handle as needed
                y_tensor = torch.zeros(
                    (2),
                    device=self.device)  # Adjust shape/content if necessary
            elif not isinstance(y_data, torch.Tensor):
                # Try converting y_data if it's not a tensor
                try:
                    y_tensor = torch.tensor(y_data, device=self.device)
                except Exception as e:
                    print(
                        f"Warning: Could not convert label from RecordingsDataset to tensor at index {idx}. Error: {e}"
                    )
                    # Fallback: maybe return a placeholder? Depends on requirements.
                    y_tensor = torch.zeros((2),
                                           device=self.device)  # Placeholder
            else:
                y_tensor = y_data.to(self.device)

            # Ensure x has channel dimension
            if x_tensor.ndim == 1 and self.channels > 0:
                x_tensor = x_tensor.unsqueeze(0)

            sample = x_tensor, y_tensor

        else:
            # Standard path using self.get_x and self.y
            x_tensor = self.get_x(idx)  # Already returns tensor on self.device

            if self.y is None:
                y_tensor = torch.zeros(
                    (2), device=self.device)  # Adjust size if needed
            else:
                y_tensor = self.y[idx]  # Already on self.device

            sample = x_tensor, y_tensor

        return sample  # (tensor, tensor) on self.device

    def __getitem__(self, idx):
        # Step 1: Get merged sample or original sample
        # These should now return (tensor, tensor) on self.device
        sample_tensor = self.get_merged_sample()
        if sample_tensor is None:
            sample_tensor = self.get_sample(
                idx)  # Gets (tensor, tensor) on self.device

        # Current state: sample_tensor = (x_tensor: Tensor, y_tensor: Tensor) on self.device

        x_current, y_current = sample_tensor  # Unpack tensors

        # Step 2: Apply batch transforms sequentially (operating on tensors)
        if 'mix_channels' in self.batch_transforms:
            # mix_channels now expects and returns (tensor, tensor) on self.device
            x_current, y_current = self.mix_channels(
                (x_current, y_current),
                mix_channels_coeff=self.mix_channels_coeff)

        if 'AddGaussianNoise' in self.batch_transforms:
            # _apply_gaussian_noise now expects and returns tensor on self.device
            x_current = self._apply_gaussian_noise(x_current)

        # Final sample tuple before optional transform
        sample_final = (x_current, y_current)  # Tensors on self.device

        # Step 3: Apply final transform
        if self.transform:
            # self.transform should expect (tensor, tensor) on self.device
            sample_final = self.transform(sample_final)
            # Ensure transform output stays on the correct device, or handle potential moves
            if isinstance(sample_final, tuple) and len(sample_final) > 1:
                if sample_final[0].device != self.device or sample_final[
                        1].device != self.device:
                    print(
                        "Warning: Transform potentially moved tensors off the specified device."
                    )
                    # Optionally move back:
                    # sample_final = (sample_final[0].to(self.device), sample_final[1].to(self.device))

        return sample_final  # Final sample (likely (tensor, tensor))

    def _apply_gaussian_noise(self, x_tensor):
        """Applies Gaussian noise augmentation directly to the input tensor."""
        # Assumes x_tensor is already a tensor on self.device with self.xdtype
        if torch.rand(1).item() < self.gauss_prob:
            # Generate amplitude (scalar)
            amplitude = torch.rand(1).item() * (
                self.gauss_max_amplitude -
                self.gauss_min_amplitude) + self.gauss_min_amplitude
            # Generate noise on the same device and add
            noise = torch.randn_like(
                x_tensor)  # Creates noise with same shape, dtype, device
            x_tensor = x_tensor + noise * amplitude
            # No need to convert dtype/device again as randn_like preserves them

        return x_tensor  # Tensor on self.device

    def mix_channels(self, sample, mix_channels_coeff):
        '''
            Mix audio channels using PyTorch tensors. Input sample is (tensor, tensor).
            Returns (tensor, tensor) on self.device.
        '''
        x_tensor, y_tensor = sample  # Unpack tensors (already on self.device)

        # x_tensor should have shape (channels, length), e.g., [2, 16000]
        if x_tensor.ndim < 2 or x_tensor.shape[
                0] < 2:  # Cannot mix if not stereo
            # If mono or unexpected shape, return as is (already tensor)
            # Ensure shape [1, length] for consistency if mono
            if x_tensor.ndim == 1:
                x_tensor = x_tensor.unsqueeze(0)  # [L] -> [1, L]
            return x_tensor, y_tensor  # Return original tensors

        # We have at least 2 channels
        # Mix channels using tensor operations
        if torch.rand(1).item() < 0.5:  # Use torch.rand for randomness
            mixed_x = x_tensor[0, :] * (
                1 - mix_channels_coeff) + x_tensor[1, :] * mix_channels_coeff
        else:
            mixed_x = x_tensor[1, :] * (
                1 - mix_channels_coeff) + x_tensor[0, :] * mix_channels_coeff
        # mixed_x is now 1D, shape (length,)
        # Reshape to (1, length) for consistency
        return mixed_x.unsqueeze(
            0), y_tensor  # Return (tensor, tensor) on self.device

    def flip_coin_for_silence(self, y, silence_index, prob=0.5):
        '''Modifies y tensor based on silence_index. Assumes y is tensor, returns tensor.'''
        # y should be a 1D tensor of labels, silence_index a scalar tensor or int
        y_modified = y.clone()  # Work on a clone (already on self.device)
        silence_present = (y_modified[silence_index] == 1
                          )  # Check if silence label is active

        if silence_present:
            # Use torch.rand for randomness
            if torch.rand(1).item() > prob:  # Make NON-SILENT
                y_modified[silence_index] = 0  # Just remove the silence label
            else:  # Make ONLY SILENT
                y_modified.zero_()  # Wipe other labels using inplace zero_
                y_modified[silence_index] = 1  # Set only silence

        return y_modified  # Return potentially modified tensor on self.device

    def get_merged_sample(self,):
        # Returns (tensor, tensor) or None, ensures tensors are on self.device
        sample = None
        merge_transforms = ['random_merge', 'random_mergev2']
        active_merge_transform = None
        num_merge_transforms = 0
        for transform in merge_transforms:
            if transform in self.batch_transforms:
                num_merge_transforms += 1
                active_merge_transform = transform

        if num_merge_transforms > 1:
            raise ValueError(
                'Only one merge transform can be applied at a time.')
        elif num_merge_transforms == 1:
            # Call the specific merge function which should now return tensors
            if active_merge_transform == 'random_mergev2':
                # Pass instance's non_associative_labels tensor
                sample = self.random_mergev2()
            elif active_merge_transform == 'random_merge':
                # Pass instance's non_associative_labels tensor
                sample = self.random_merge()
            # Ensure the returned sample (if not None) has tensors on the correct device
            if sample is not None:
                x, y = sample
                if x.device != self.device or y.device != self.device:
                    print(
                        f"Warning: Merge function {active_merge_transform} returned tensors on wrong device. Moving."
                    )
                    sample = (x.to(self.device), y.to(self.device))

        return sample  # (tensor, tensor) on self.device, or None

    def merge_samples(self, id_1, id_2):
        # Merges samples identified by id_1, id_2 using PyTorch tensors.
        # Uses self.non_associative_labels (already a tensor on self.device).
        # Returns (tensor, tensor) on self.device.

        # get_x and self.y[id] should provide tensors on self.device
        left_x = self.get_x(id_1)
        right_x = self.get_x(id_2)

        left_y = self.y[id_1]
        right_y = self.y[id_2]

        # Ensure compatible shapes for merging (especially length)
        # Simple truncation/padding might be needed in real scenarios if lengths differ
        min_len = min(left_x.shape[-1], right_x.shape[-1])
        if left_x.shape[-1] != right_x.shape[-1]:
            print(
                f"Warning: Merging samples with different lengths ({left_x.shape[-1]} vs {right_x.shape[-1]}). Truncating."
            )
            left_x = left_x[..., :min_len]
            right_x = right_x[..., :min_len]

        # Generate random percentage using torch.rand
        percentage = torch.rand(
            1, device=self.device).item() * (0.85 - 0.15) + 0.15  # Scalar value

        # Perform merge using tensor operations
        x_merged = (left_x * (1 - percentage)) + (right_x * percentage)
        # x_merged automatically inherits dtype and device

        # Merge labels (assuming binary labels {0, 1})
        y_merged = left_y + right_y
        # Clamp values to 1.0 (use torch.clamp) - ensure y tensors are float for this if needed
        y_merged = torch.clamp(y_merged.float(), max=1.0).to(
            left_y.dtype)  # Convert back to original dtype

        # Handle non-associative labels (e.g., silence)
        if self.non_associative_labels is not None and len(
                self.non_associative_labels) > 0:
            silence_index = self.non_associative_labels[
                0]  # Assuming first is silence index
            # Pass the current merged y tensor, get modified version back
            y_merged_modified = self.flip_coin_for_silence(y_merged,
                                                           silence_index,
                                                           prob=0.5)

            # Check if the *resulting* y IS silence
            is_only_silence = (y_merged_modified[silence_index]
                               == 1) and (torch.sum(y_merged_modified) == 1)

            if is_only_silence:
                # If only one input was silent, use its x.
                left_was_silent = (left_y[silence_index] == 1)
                right_was_silent = (right_y[silence_index] == 1)
                if not (left_was_silent and
                        right_was_silent):  # If not both were silent originally
                    x_merged = left_x if left_was_silent else right_x
                    # x_merged is already the correct tensor

            # Update y_merged with the result from flip_coin
            y_merged = y_merged_modified

        return x_merged, y_merged  # Returns (tensor, tensor) on self.device

    def random_merge(self, merge_probability=0.5):
        '''
            Randomly pick two samples and merge them, with replacement.
            Returns (tensor, tensor) on self.device.
            Uses self.non_associative_labels (instance attribute).
        '''
        # Decide whether to merge based on probability
        # Use torch.rand() for comparison
        if torch.rand(1).item() >= merge_probability:
            # Don't merge, return a single random sample
            random_id = random.randint(0, self.__len__() - 1)
            return self.get_sample(random_id)  # Returns (tensor, tensor)

        # Proceed with merging
        random_id_1 = random.randint(0, self.__len__() - 1)
        random_id_2 = random.randint(0, self.__len__() - 1)

        # merge_samples now uses instance attribute self.non_associative_labels implicitly
        # and returns (tensor, tensor) on self.device
        x, y = self.merge_samples(random_id_1, random_id_2)
        return x, y

    def shuffle_indexes(self,):
        self.id_mix = list(range(self.__len__()))
        random.shuffle(self.id_mix)  # Keep using random.shuffle for index lists

    def random_mergev2(self, merge_probability=0.5):
        '''
            Randomly pick two samples and merge them, without replacement from self.id_mix.
            Merging happens with merge_probability.
            Returns (tensor, tensor) on self.device.
            Uses self.non_associative_labels (instance attribute).
        '''
        if len(self.id_mix) < 1:
            self.shuffle_indexes()
            if not self.id_mix:  # Handle case where dataset len is 0
                raise IndexError("Cannot get sample from empty dataset.")

        # Decide whether to merge based on probability
        # Use torch.rand() for comparison
        if torch.rand(1).item() >= merge_probability:
            # Don't merge, return a single random sample from id_mix
            random_id = self.id_mix.pop()
            return self.get_sample(random_id)  # Returns (tensor, tensor)

        # Proceed with merging, need at least 2 ids
        if len(self.id_mix) < 2:
            # Not enough IDs left for merging, shuffle and check again
            self.shuffle_indexes()
            if len(
                    self.id_mix
            ) < 2:  # Still not enough (dataset size < 2 or just shuffled to < 2)
                # Fallback: return a single sample instead of merging
                print(
                    "Warning: Not enough unique samples left for random_mergev2, returning single sample."
                )
                if not self.id_mix:  # If shuffle resulted in empty list (len=0 dataset)
                    raise IndexError(
                        "Cannot get sample from empty dataset after shuffle.")
                random_id = self.id_mix.pop()
                return self.get_sample(random_id)  # Returns (tensor, tensor)

        # Pop two IDs for merging
        random_id_1 = self.id_mix.pop()
        random_id_2 = self.id_mix.pop()

        # merge_samples now uses instance attribute self.non_associative_labels implicitly
        # and returns (tensor, tensor) on self.device
        x, y = self.merge_samples(random_id_1, random_id_2)
        return x, y
