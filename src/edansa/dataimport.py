'''Module handling importing data from external sources.

'''
# from argparse import ArgumentError
from copy import deepcopy
from typing import Union, Optional, List, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import os
import traceback

from datetime import datetime
import csv

from pathlib import Path
from collections.abc import MutableMapping
from collections import Counter

import yaml
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import Dataset as TorchDataset

import edansa.taxoutils
import edansa.utils
import edansa.clippingutils
import edansa.audio
import edansa.modelarchs

from edansa.io import TIMESTAMP_DATASET_FORMAT, LENGTH_DATASET_FORMAT
from edansa.io import AudioRecordingFiles

import logging  # Import logging


class Audio():
    """Single audio sample within the dataset.
    """

    def __init__(
        self,
        path: Union[str, Path],
        length_seconds: float = -1,
        sampling_rate: int = -1,
        taxo_codes: Union[List[str], None] = None,
        clipping: Optional[Any] = None,  # Updated type hint
        # shape = (# of 10 seconds,number of channels)
        location=None,
        region=None,
        data_version=None,
        annotator=None,
        comments=None,
        original_recording_path=None,
        timestamps=None,
    ):
        self.path = Path(path)
        self.name = self.path.name
        self.suffix = self.path.suffix
        self.clipping = clipping
        self.data = np.empty(0)  # suppose to be np.array
        self.sr: Optional[int] = None  # sampling rate
        self.location = location
        self.region = region
        self.samples = []
        self.data_version = data_version
        self.annotator = annotator
        self.comments = comments
        self.original_recording_path = original_recording_path
        self.timestamps = timestamps  # (start,end)
        self.length = length_seconds  # in seconds
        self.sampling_rate = sampling_rate
        self.taxo_codes = taxo_codes
        self.extra_columns: dict = {}  # Initialize dict for extra columns

    def __str__(self,):
        return str(self.name)

    def __repr__(self,):
        return f'{self.path}, length:{self.length}'

    def pick_channel_by_clipping(self, excerpt_length):
        # Restore tensor handling logic
        is_tensor = False
        if isinstance(self.data, torch.Tensor):
            is_tensor = True
            self.data = self.data.numpy()

        if len(self.data.shape) == 1 or self.data.shape[0] == 1:
            # If it was a tensor, convert back just in case (though no change occurred)
            if is_tensor:
                self.data = torch.from_numpy(self.data)
            return None  # Already single channel or clipping not applicable
        if self.clipping is None:
            # Convert back before raising error if it was a tensor
            if is_tensor:
                self.data = torch.from_numpy(self.data)
            raise ValueError(f'{self.path} does not have clipping information.')

        cleaner_channel_indexes = np.argmin(self.clipping, axis=1)
        new_data = np.empty(self.data.shape[-1], dtype=self.data.dtype)

        excpert_len_jump = self.sr * excerpt_length
        for ch_i, data_i in zip(cleaner_channel_indexes,
                                range(0, self.data.shape[-1],
                                      excpert_len_jump)):
            new_data[data_i:data_i +
                     excpert_len_jump] = self.data[ch_i, data_i:data_i +
                                                   excpert_len_jump]

        self.data = new_data[:]

        # Convert back to tensor if it was originally
        if is_tensor:
            self.data = torch.from_numpy(self.data)

        return 1

    def select_channel(self, method, excerpt_length):
        """Selects a single channel from multi-channel audio based on the method.

        Modifies self.data in place to be a 1D array.

        Args:
            method: 'clipping', 'average', 'channel_N'
            excerpt_length: Needed for 'clipping' method.
        """
        # If data is already 1D, no action needed
        if len(self.data.shape) == 1:
            return

        # Ensure data is numpy array for processing
        is_tensor = False
        if isinstance(self.data, torch.Tensor):
            is_tensor = True
            self.data = self.data.numpy()

        if method == 'clipping':
            if hasattr(self, 'clipping') and self.clipping is not None:
                self.pick_channel_by_clipping(excerpt_length)
            else:
                # Use logging
                logging.warning(
                    f"No clipping info available for {self.path}. Using first channel."
                )
                self.data = self.data[0]
        elif method == 'average':
            self.data = self.data.mean(axis=0)
        elif method.startswith('channel_'):
            try:
                channel_idx = int(method.split('_')[1])
                if channel_idx < self.data.shape[0]:
                    self.data = self.data[channel_idx]
                else:
                    # Use logging
                    logging.warning(
                        f"Requested channel {channel_idx} exceeds available channels ({self.data.shape[0]}) for {self.path}. Using first channel."
                    )
                    self.data = self.data[0]
            except (IndexError, ValueError):
                # Use logging
                logging.warning(
                    f"Invalid channel specification for {self.path}. Using first channel."
                )
                self.data = self.data[0]
        else:
            # Use logging
            logging.warning(
                f"Unknown channel selection method '{method}' for {self.path}. Using first channel."
            )
            self.data = self.data[0]

        # Convert back to tensor if it was originally
        if is_tensor:
            self.data = torch.from_numpy(self.data)

    def load_data(
        self,
        dtype: torch.dtype = torch.float32,
        store=True,
        resample_rate=-1,
        backend=None,
        mono=False,
        normalize: bool = True,
        channels_first: bool = True,
    ):

        sound_array, sr = edansa.audio.load(
            self.path,
            dtype=dtype,
            resample_rate=resample_rate,
            backend=backend,
            mono=mono,
            normalize=normalize,
            channels_first=channels_first,
        )

        # Calculate and update duration based on loaded data
        if sr is not None and sr > 0:
            num_samples = sound_array.shape[-1]
            self.length = num_samples / sr
        else:
            # Keep the initialized value (e.g., -1) or set to a specific error value if sr is invalid
            logging.warning(
                f"Invalid sample rate ({sr}) obtained for {self.path}. Cannot calculate duration."
            )
            # self.length remains whatever it was initialized to, or you could set self.length = -1 here explicitly

        if store:
            self.data = sound_array
            self.sr = sr
            return self.data, self.sr
        else:
            return sound_array, sr

    def unload(self):
        self.unload_data()
        self.unload_samples()

    def unload_data(self):
        del self.data
        self.data = torch.empty(0)

    def unload_samples(self):
        del self.samples
        self.samples = []

    def get_data_by_value(self,
                          dtype=torch.float32,
                          resample_rate=-1,
                          **kwargs):
        if (self.data.size is None) or (self.data.size == 0):
            return self.load_data(
                dtype=dtype,  # type: ignore
                store=False,
                resample_rate=resample_rate,
                **kwargs)
        else:
            if kwargs.get('mono', False) and self.channel_count() > 1:
                print(
                    f"Warning: get_data_by_value returning cached stereo data for {self.path} despite mono=True request."
                )
            return self.data[:], self.sr

    def load_info(self,
                  row,
                  excell_names2code=None,
                  version='V2',
                  dataset_folder=None):
        """_summary_

        Args:
            row (dict): row of the dataset, requires
                            'length',
        """
        # Define known/handled keys (lowercase)
        known_keys_lower = {
            'length',
            'duration_sec',
            'start_date_time',
            'end_date_time',
            'clip_path',
            'clip path',
            'region',
            'location',
            'comments',
            'annotator',
            'data_version',
            'file_name'  # Add any other specifically handled keys
        }

        # Add keys from excell_names2code (if provided) to known keys
        if excell_names2code:
            known_keys_lower.update(k.lower() for k in excell_names2code.keys())

        # Store extra columns
        self.extra_columns = {}
        for key, value in row.items():
            if key.lower() not in known_keys_lower:
                self.extra_columns[key] = value

        del version
        # lenght can be duration_sec or length or Length
        length = row.get('length', row.get('Length', row.get('duration_sec')))
        times_str_start = row.get('start_date_time')
        times_str_end = row.get('end_date_time')

        start_time = datetime.strptime(times_str_start,
                                       TIMESTAMP_DATASET_FORMAT)
        end_time = datetime.strptime(times_str_end, TIMESTAMP_DATASET_FORMAT)

        try:
            # Try to parse as ISO 8601 format
            length_time = datetime.strptime(length, LENGTH_DATASET_FORMAT)
            total_seconds = (length_time -
                             datetime(year=length_time.year,
                                      month=length_time.month,
                                      day=length_time.day)).total_seconds()
        except ValueError:
            try:
                # If parsing as ISO 8601 format fails, try to parse as float
                total_seconds = float(length)
            except ValueError:
                print(
                    f"Invalid format for {length}, expected ISO 8601 format or float"
                )
                total_seconds = None

        clip_path = row.get('clip_path', row.get('Clip Path'))

        if dataset_folder is not None:
            dataset_folder = Path(dataset_folder)
            self.path = dataset_folder / clip_path
        else:
            self.path = Path(clip_path)

        # info = torchaudio.info(self.path)
        # self.length = info.num_frames / info.sample_rate
        # self.sr = info.sample_rate
        # self.num_frames = info.num_frames
        # self.num_channels = info.num_channels

        self.name = self.path.name
        self.suffix = self.path.suffix
        self.length = float(total_seconds)
        self.location = row.get('location', row.get('Site ID',
                                                    None)).strip().lower()
        self.region = row.get('region', None).strip().lower()
        self.data_version = row.get('data_version', None)
        self.annotator = row.get('annotator', row.get('Annotator', ''))
        self.comments = row.get('comments', row.get('Comments', ''))
        self.original_recording_path = row.get('file_name',
                                               row.get('File Name', None))
        self.timestamps = (start_time, end_time)

        if excell_names2code is not None:
            self.headers2taxo_code(row, excell_names2code)

    def headers2taxo_code(self, row, excell_names2code):
        self.taxo_y = {}
        for header_name, taxo_code in excell_names2code.items():
            # value is 1 or 0
            value = row.get(header_name, None)
            if value is not None:
                if value == '':
                    continue
                    # print('Warning: empty value for ', header_name, taxo_code)
                try:
                    self.taxo_y[taxo_code] = float(value)
                except ValueError:
                    print('Warning: could not convert to float ', value, 'for',
                          header_name, taxo_code)
        self.taxo_codes = list(self.taxo_y.keys())
        return self.taxo_y, self.taxo_codes

    def sample_count(
        self,
        excerpt_length=10,
        sample_length_limit=2,
    ):
        '''count number of samples can be extracted from the audio file
        '''
        if self.length == -1 or self.length is None:
            return -1

        if self.length >= sample_length_limit:
            sample_count = self.length // excerpt_length
            left_overs = (self.length % excerpt_length)
            if left_overs > sample_length_limit and left_overs > 0:
                sample_count += 1
        else:
            sample_count = 0

        return sample_count

    def data_to_samples(self, excerpt_len=10):
        '''
            self.data is a numpy array of shape (channels,n_samples)
            excerpt_len is length of excerpt in seconds
        '''

        self.samples = self.divide_long_sample(excerpt_len=excerpt_len,)

    def channel_count(self, data=None):
        data = self.data if data is None else data
        return data.shape[0] if len(data.shape) > 1 else 1

    def divide_long_sample(self,
                           data=None,
                           sr=None,
                           excerpt_len=10,
                           min_sample_len=5):
        """
        Divide the long audio file into chunks of size excerpt_len using PyTorch

        Assumes (channel, time) shape of data, (time,) is also acceptable.

        Args:
            data (torch.Tensor): Audio data
            sr (int): Sampling rate
            excerpt_len (int): Length of each excerpt in seconds
            min_sample_len (int): Minimum length of each excerpt in seconds
        Returns:
            list: List of audio excerpts
        """
        data = data if data is not None else self.data
        sr = sr if sr is not None else self.sr

        if data is None or sr is None:
            raise ValueError(f'data or sr is None, {sr=}, {data.shape=}')

        ch_count = self.channel_count(data)
        if ch_count == 1:
            data = data.reshape(1, -1)
        elif ch_count > 2:
            raise ValueError(f'ch_count > 2, {ch_count=}')

        excerpt_sample_size = int(excerpt_len * sr)
        numpy_flag = False
        if not torch.is_tensor(data):
            numpy_flag = True
            data = torch.from_numpy(data)

        # Split the data using torch.split
        samples = list(
            torch.split(
                data,  # type:ignore
                excerpt_sample_size,
                dim=1))

        # Handle the leftover samples and padding
        if samples[-1].shape[1] < min_sample_len * sr:
            samples.pop()  # remove if shorter than min_sample_len
        elif samples[-1].shape[1] < excerpt_sample_size:
            padding = torch.zeros(
                (ch_count, excerpt_sample_size - samples[-1].shape[1]))
            padded_data = torch.cat([samples[-1], padding], dim=1)
            samples[-1] = padded_data

        # Convert samples back to numpy arrays
        if numpy_flag:
            samples = [sample.numpy() for sample in samples]

        if ch_count == 1:
            samples = [sample.reshape(-1) for sample in samples]

        return samples

    def get_row_format(self):
        assert self.timestamps is not None, 'timestamps is not set'
        assert len(self.timestamps) == 2, 'timestamps should be a tuple of 2'
        time_diff = (self.timestamps[1] - self.timestamps[0])
        time_diff_total_seconds = time_diff.total_seconds()
        time_diff = datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0) + time_diff

        if self.length != time_diff_total_seconds:
            print('Warning: length != (end-start),\n' +
                  f'{self.length} != {time_diff_total_seconds}')
        length_str = datetime.strftime(time_diff, LENGTH_DATASET_FORMAT)

        # Start with the explicitly handled/modified columns
        output_row = {
            'data_version':
                self.data_version,
            'region':
                self.region,
            'location':
                self.location,
            'comments':
                self.comments,
            'start_date_time':
                datetime.strftime(self.timestamps[0], TIMESTAMP_DATASET_FORMAT),
            'end_date_time':
                datetime.strftime(self.timestamps[1], TIMESTAMP_DATASET_FORMAT),
            'duration_sec':
                length_str,
            'clip_path':
                str(self.path),
            # Add back other core fields if they were stored separately on self
            'annotator':
                self.annotator,
            'file_name':
                self.original_recording_path,
        }

        # Update the output row with all the stored extra columns
        # This will overwrite any keys that might have been in both (e.g., if region was also in extra_columns)
        # but it prioritizes the explicitly set values in output_row
        print(f"DEBUG: extra_columns before update: {self.extra_columns}"
             )  # DEBUG
        output_row.update(self.extra_columns)

        # Replace original path-related columns from extra_columns with the new path
        # Handle potential case variations ('Clip Path', 'clip_path', 'file_path', etc.)
        output_row['clip_path'] = str(self.path)  # Ensure the new path is used
        if 'Clip Path' in output_row:  # Remove old capitalized version if present
            del output_row['Clip Path']
        # Decide if original 'file_path' should be kept, removed, or updated
        # If 'file_name' is meant to be the original file, keep it as is from output_row['file_name']
        # If 'file_path' (from extra_columns) should be removed, uncomment below:
        # if 'file_path' in output_row:
        #     del output_row['file_path']

        # Check for None values *after* merging
        if None in output_row.values():
            print('Warning: some values in the generated row are None')
            print(output_row)

        print(f"DEBUG: final output_row for CSV: {output_row}")  # DEBUG
        return output_row


class Dataset(MutableMapping):
    """A dictionary that holds data points."""

    def __init__(
        self,
        csv_path_or_rows=None,
        dataset_name_v='',
        excerpt_len=10,
        min_allowed_sample_length=2,
        dataset_cache_folder='',
        dataset_folder=None,
        data_dict=None,
        excell_names2code=None,
        taxonomy_file_path=None,
        target_taxo=None,
    ):
        self.store = dict()
        if data_dict is not None:
            self.update(dict(**data_dict))  # use the free update to set keys
        self.excerpt_length = excerpt_len  # in seconds
        self.min_allowed_sample_length = min_allowed_sample_length
        self.dataset_name_v = dataset_name_v
        self.excell_names2code = excell_names2code
        self.csv_path_or_rows = csv_path_or_rows
        self.target_taxo = target_taxo

        if dataset_cache_folder == '':
            self.dataset_cache_folder = ''
        else:
            self.dataset_cache_folder = Path(dataset_cache_folder)
        self.dataset_folder = dataset_folder
        if csv_path_or_rows is not None:
            self.load_csv(self.csv_path_or_rows,
                          self.dataset_name_v,
                          self.dataset_cache_folder,
                          excell_names2code=self.excell_names2code)
        if taxonomy_file_path is not None:
            self.load_taxonomyfile(taxonomy_file_path)

    def __getitem__(self, key):
        return self.store[self._keytransform(key)]

    def __setitem__(self, key, value):
        self.store[self._keytransform(key)] = value

    def __delitem__(self, key):
        del self.store[self._keytransform(key)]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def _keytransform(self, key):
        return key

    def load_taxonomyfile(self, taxonomy_file_path):
        # Store taxonomy information in the dataset.
        # Taxonomy file
        taxonomy_file_path = Path(taxonomy_file_path)
        with open(taxonomy_file_path, encoding='utf-8') as f:
            taxonomy = yaml.load(f, Loader=yaml.FullLoader)

        self.taxonomy = edansa.taxoutils.Taxonomy(deepcopy(taxonomy))

    def load_csv(self,
                 csv_path_or_rows,
                 dataset_name_v='',
                 dataset_cache_folder: Union[str, Path] = '',
                 dataset_folder=None,
                 excell_names2code=None):
        """read path, len of megan labeled files from csv file, (lnength col.)
        store them in a dataimport.dataset, keys are gonna be sample file path
        """
        if dataset_folder is None:
            dataset_folder = self.dataset_folder
        # if excell_names2code is None and self.excell_names2code is None:
        #     raise ArgumentError('excell_names2code shoudl be provided')
        if isinstance(csv_path_or_rows, str):
            dataset_rows = self._read_csv_file(csv_path_or_rows)
        else:
            dataset_rows = csv_path_or_rows[:]
        # self.store = edansa.dataimport.Dataset()
        for row in dataset_rows:
            row_key = row.get('clip_path', row.get('Clip Path', None))
            self.store[row_key] = Audio('', -1)
            self.store[row_key].load_info(row,
                                          excell_names2code=excell_names2code,
                                          dataset_folder=dataset_folder)

        # path has to be inque but is file names are unique ?

        # assert len(set([i.name for i in self.store.values()
        #    ])) == len(list(self.store.keys()))

        print('dataset_cache_folder is ', dataset_cache_folder)
        self.dataset_name_v = dataset_name_v
        self.dataset_cache_folder = dataset_cache_folder

        return True

    def _read_csv_file(self, csv_path):
        with open(csv_path, encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            reader = list(reader)
            reader_strip = []
            for row in reader:
                row = {r: row[r].strip() for r in row}
                reader_strip.append(row)
            reader = reader_strip.copy()
        return reader

    def export_csv(self, csv_path):
        """export dataset to csv file"""
        rows = []
        for sample in self.store.values():
            rows.append(sample.get_row_format())

        # Remove explicit fieldnames; let utils.write_csv infer from all rows
        # fieldnames = edansa.taxoutils.excell_all_headers
        edansa.utils.write_csv(csv_path, rows, fieldnames=None)

        return True

    def dataset_clipping_percentage(self,
                                    output_folder: Union[str, Path,
                                                         None] = None) -> tuple:
        """Calculate clipping info for the dataset.

        If output_folder is provided, checks for cached results and saves new results.
        If output_folder is None, calculates clipping but does not cache.
        
        Args:
            output_folder: Path to save/load cached results. If None, no caching.

        Returns:
            Tuple (dict, list): Dictionary of clipping results and list of errors.
        """
        dict_output_path = None
        clipping_error_path = None
        cache_found = False

        if output_folder is not None and self.dataset_name_v:
            output_folder = Path(output_folder)
            dict_output_path = output_folder / (self.dataset_name_v +
                                                '_1,0.pkl')
            clipping_error_path = output_folder / (self.dataset_name_v +
                                                   '_1,0_error.pkl.pkl')
            if dict_output_path.exists():
                cache_found = True

        # Load from cache if found
        if cache_found and dict_output_path:
            print(f"Loading cached clipping info from {dict_output_path}")
            clipping_results = np.load(dict_output_path, allow_pickle=True)[()]
            if clipping_error_path and clipping_error_path.exists():
                clipping_errors = np.load(clipping_error_path,
                                          allow_pickle=True)[()]
            else:
                clipping_errors = []
            return clipping_results, clipping_errors

        # Calculate if no cache or no output folder specified
        else:
            # if output_folder:
            #     print(
            #         f'Could not find clipping info cache at {dict_output_path}. Calculating.'
            #     )
            # else:
            #     print(
            #         'No output_folder provided. Calculating clipping info without caching.'
            #     )

            path_list = [value.path for value in self.store.values()]

            # Determine if we should save results (only if output_folder is provided)
            should_save = output_folder is not None

            (all_results_dict,
             files_w_errors) = edansa.clippingutils.run_task_save(
                 path_list,
                 self.dataset_name_v,
                 output_folder,  # Can be None 
                 1.0,
                 save=should_save)
            return all_results_dict, files_w_errors

    def update_samples_w_clipping_info(self,
                                       output_folder: Union[str, Path,
                                                            None] = None):
        """Updates Audio samples in the dataset with clipping info.
        
        Args:
            output_folder: Path to save/load cached clipping results. If None, no caching.
        """
        all_results_dict, files_w_errors = self.dataset_clipping_percentage(
            output_folder)
        del files_w_errors  # Currently not handling errors further here
        for sample in self.store.values():
            clipping = all_results_dict.get(str(sample.path), None)
            if clipping is not None:
                sample.clipping = clipping

    def load_single_audio_files(self, key, value, cached_dict, **kwargs):
        data = cached_dict.get(str(value.path), None)
        if data is None:
            sound_array, sr = value.load_data(**kwargs)
        else:
            sound_array, sr = data
            self.store[key].data = sound_array
            self.store[key].sr = sr
        return key

    def load_audio_files(
        self,
        cached_dict_path=None,
        dtype=torch.float32,
        resample_rate=48000,
        mono=True,
        channels_first=True,
        max_workers=10,
        use_threads=False,
    ):
        """load audio files from disk, if cached_dict_path is provided, load
        from cache
        """

        if cached_dict_path is not None and cached_dict_path:
            print(f'loading from cache at {cached_dict_path}')
            cached_dict = np.load(cached_dict_path, allow_pickle=True)[()]
        else:
            print('no cache for audio files found ' +
                  'loading original files to memory')
            cached_dict = {}

        # If use_threads is set to True, load using threads
        if use_threads:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self.load_single_audio_files,
                        key,
                        value,
                        cached_dict,
                        dtype=dtype,
                        backend=None,
                        store=True,
                        resample_rate=resample_rate,
                        mono=mono,
                        channels_first=channels_first,
                    ):
                        key for key, value in self.store.items()
                }
                # Progress bar initialization
                pbar = tqdm(total=len(futures),
                            desc='Loading audio files',
                            dynamic_ncols=True)

                for future in as_completed(futures):
                    key = futures[future]
                    try:
                        future.result()
                        pbar.update(1)
                    except Exception as exc:
                        print(f'{key} generated an exception: {exc}')

                pbar.close()
        # Otherwise, load serially without threads
        else:
            pbar = tqdm(total=len(self.store.items()),
                        desc='Loading audio files',
                        dynamic_ncols=True)

            for key, value in self.store.items():
                try:
                    self.load_single_audio_files(
                        key,
                        value,
                        cached_dict,
                        dtype=dtype,
                        backend=None,
                        store=True,
                        resample_rate=resample_rate,
                        mono=mono,
                        channels_first=channels_first,
                    )
                    pbar.update(1)
                except Exception as exc:
                    print(f'{key} generated an exception: {exc}')

            pbar.close()

    def pick_channel_by_clipping(self):
        for _, audio_sample in self.store.items():
            if audio_sample.clipping is None:
                print('Found audio samples with no clipping info, loading')
                self.update_samples_w_clipping_info()

        for _, audio_sample in self.store.items():
            # this is a function of Audio
            audio_sample.pick_channel_by_clipping(self.excerpt_length)

    def create_cache_pkl(self, output_file_path):
        '''save data files of samples as pkl.
        '''
        data_dict = {}
        if Path(output_file_path).exists():
            raise ValueError(f'{output_file_path} already exists')
        for value in self.store.values():
            data_dict[str(value.path)] = value.data, value.sr

        with open(output_file_path, 'wb') as f:
            np.save(f, data_dict)  # type: ignore

    def count_samples_per_taxo_code(self,
                                    sample_length_limit=None,
                                    version='V2'):
        """Go through rows of the excell and count category population

            returns:
                taxo_code_counter: {'0.0.1':12,...}
        """
        no_taxo_code = []
        taxo_code_counter = Counter()
        if sample_length_limit is None:
            sample_length_limit = self.min_allowed_sample_length

        for audio_ins in self.store.values():
            if audio_ins.taxo_codes is None:
                no_taxo_code.append(audio_ins)
                continue
            sample_count = audio_ins.sample_count(
                excerpt_length=self.excerpt_length,
                sample_length_limit=sample_length_limit)

            if version == 'V1':
                taxo_code_counter.update({audio_ins.taxo_codes: sample_count})
            elif version == 'V2':
                taxo_code_counter.update({
                    taxo_code: sample_count
                    for taxo_code in audio_ins.taxo_codes
                })
        if no_taxo_code:
            msg = 'following samples do not have taxonomy info:'
            print(msg)
            for i in no_taxo_code:
                print(i)
        return taxo_code_counter

    def count_samples_per_location_by_taxo_code(self,
                                                sample_length_limit=None,
                                                version='V2'):
        """Go through rows of the excell and count category population

            returns:
                taxo_code_counter: {'0.0.1':{'location_1':12,
                'location_1':45 ...} ...}
        """
        no_taxo_code = []
        taxo2loc_dict = {}

        if sample_length_limit is None:
            sample_length_limit = self.min_allowed_sample_length

        for audio_ins in self.store.values():
            if audio_ins.taxo_codes is None:
                no_taxo_code.append(audio_ins)
                continue

            sample_count = audio_ins.sample_count(
                excerpt_length=self.excerpt_length,
                sample_length_limit=sample_length_limit)

            if version == 'V1':
                taxo2loc_dict.setdefault(audio_ins.taxo_codes, Counter({}))

                taxo2loc_dict[audio_ins.taxo_codes] = taxo2loc_dict[
                    audio_ins.taxo_codes] + Counter(
                        {audio_ins.location: sample_count})

            elif version == 'V2':
                for taxonomy_code in audio_ins.taxo_codes:
                    taxo2loc_dict.setdefault(taxonomy_code, Counter({}))

                    taxo2loc_dict[taxonomy_code] = taxo2loc_dict[
                        taxonomy_code] + Counter({
                            '::'.join([audio_ins.region, audio_ins.location]):
                                sample_count
                        })
        if no_taxo_code:
            msg = 'following samples do not have taxonomy info:'
            print(msg)
            for i in no_taxo_code:
                print(i)
        return taxo2loc_dict

    def dataset_generate_samples(self, excerpt_len):
        '''divida into chunks by expected_len seconds.
            Repeats data if smaller than expected_len.

        '''
        for sound_ins in self.values():
            print(sound_ins)
            if len(sound_ins.data.shape) == 2:
                # data_to_samples(sound_ins, excerpt_len)
                sound_ins.data_to_samples(excerpt_len=excerpt_len)
            elif len(sound_ins.data.shape) == 1:
                sound_ins.data = sound_ins.data.reshape(1, -1)
                sound_ins.data_to_samples(excerpt_len=excerpt_len)
                sound_ins.data = sound_ins.data.reshape(-1)
            else:
                raise Exception('data shape not supported')
        return self


# Function to get the next item from the iterator or restart if exhausted
def get_next_item(iterator, a_dict=None):
    try:
        return next(iterator)
    except StopIteration:
        # Reinitialize the iterator once it's exhausted
        if a_dict is not None and type(a_dict) is dict:
            iterator = iter(a_dict.items())
        elif a_dict is not None:
            iterator = iter(a_dict)
        return next(iterator)


def shuffle_dict(a_dict):
    items = list(a_dict.items())
    random.shuffle(items)
    a_dict = dict(items)
    return a_dict


class RecordingsDataset(TorchDataset):
    """Dataset for loading audio files and their corresponding labels."""

    def __init__(
        self,
        recordings: Union[str, AudioRecordingFiles],
        in_memory_sample_count: int,
        weather_data: Union[str, pd.DataFrame],
        regloc_list: List[Tuple[str, str]],
        excerpt_len=10,
        resample_rate=48000,
        mono=True,
        channels_first=True,
    ):
        self.regloc_list = regloc_list
        self.recordings = self._load_recordings(recordings)
        self.in_memory_sample_count = in_memory_sample_count
        self.weather_data = self._load_weather_data(weather_data)
        self.excerpt_len = excerpt_len
        self.resample_rate = resample_rate
        self.mono = mono
        self.channels_first = channels_first
        self.data = []
        self.current_sample = 0

        self.recording2weather = self.matchrecordings2weather(
            recordings=self.recordings, weather=self.weather_data)
        self.remove_recordings_without_weather()

        assert len(self.recording2weather) > 0, 'no matching weather data'
        self.recording2weather = shuffle_dict(self.recording2weather)
        self.recording2weather_iter = iter(self.recording2weather.items())
        if self.current_sample >= len(self.data):
            self._load_data()

    def _load_recordings(self, recordings):
        if isinstance(recordings, str):
            # do not load if recordings empty string
            recordings = AudioRecordingFiles(recordings, version='v2')
        elif isinstance(recordings, AudioRecordingFiles):
            pass
        else:
            raise Exception('recordings should be str or AudioRecordingFiles')
        if self.regloc_list:
            recordings.filter_by_regloc(regloc_list=self.regloc_list,
                                        inplace=True)
        assert len(recordings) > 0, 'no recordings found'
        return recordings

    def _load_weather_data(self, weather_data):

        if isinstance(weather_data, str):
            weather = pd.read_csv(weather_data)
        elif isinstance(weather_data, pd.DataFrame):
            weather = weather_data
        else:
            raise Exception('weather_data should be str or pd.DataFrame')
        return weather

    def _load_data(self, filecount2load_inparallel=10):
        """Load the next batch of data into memory."""

        # Define a helper function for the task that should be parallelized
        def load_recording(recording_data):
            recording_path, recs_weather = recording_data
            audio = Audio(recording_path)
            try:
                audio.load_data(
                    store=True,
                    resample_rate=self.resample_rate,
                    backend=None,
                    mono=self.mono,
                    normalize=True,
                    channels_first=self.channels_first,
                )
            except Exception as e:
                print(f"Exception occurred: {type(e).__name__}, {str(e)}")
                traceback.print_exc()
                return []
            samples = audio.divide_long_sample(excerpt_len=self.excerpt_len)
            y_values = self.extract_weather_values_for_clip(
                recording_path, recs_weather, self.recordings.files)

            return [(sample, label.reshape((1,)))
                    for sample, label in zip(samples, y_values)]

        # Initialize empty data list
        self.data = []
        samples_loaded = 0

        while samples_loaded < self.in_memory_sample_count:

            # Get the next batch of recordings
            recording_data = [
                get_next_item(self.recording2weather_iter)
                for _ in range(filecount2load_inparallel)
            ]

            # Use a ThreadPool to parallelize the loading process
            with ThreadPoolExecutor() as executor:
                results = list(executor.map(load_recording, recording_data))

            # Flatten the list of results and append to self.data
            self.data.extend([item for sublist in results for item in sublist])

            samples_loaded += sum(len(sublist) for sublist in results)

        random.shuffle(self.data)

    def __len__(self):
        # return len(self.data)
        return self.in_memory_sample_count

    def __getitem__(self, idx):
        del idx
        if self.current_sample >= len(self.data):
            self._load_data()
            self.current_sample = 0
        sample, label = self.data[self.current_sample]
        self.current_sample += 1

        return sample, label

    def matchrecordings2weather(self, recordings, weather):
        """Match audio recording files to weather data."""
        start, end = recordings.get_time_period()
        weather_data = weather[(weather['start_date_time'] <= end) &
                               (weather['end_date_time'] >= start)]

        recording_groups = recordings.groupby_regloc()
        weather_groups = weather_data.groupby(['region', 'location'])

        def get_matching_weather(weather_group, recording):
            """Get matching weather data for a recording."""
            return weather_group[
                (weather_group['start_date_time'] < recording['end_date_time'])
                &
                (weather_group['end_date_time'] > recording['start_date_time'])]

        recording2weather = {}
        for (region, location), rec_group in recording_groups:
            if (region, location) not in weather_groups.groups:
                continue

            weather_group = weather_groups.get_group((region, location))
            for _, recording in rec_group.iterrows():
                matching_weather = get_matching_weather(weather_group,
                                                        recording)
                key = str(recording['clip_path'])

                if not matching_weather.empty:
                    recording2weather[key] = matching_weather.copy()
                else:
                    print(f'WARNING: No weather data for {key}')
                    recording2weather[key] = None

        return recording2weather

    def remove_recordings_without_weather(self):
        """Remove recordings without weather data."""
        self.recordings.files = self.recordings.files[
            self.recordings.files.index.isin(
                self.recording2weather.keys())].copy()
        assert len(
            self.recordings) > 0, 'no recordings left after weather filtering'
        return self.recordings

    @staticmethod
    def expand_to_intervals(row,
                            clip_start,
                            clip_end,
                            excerpt_length=10,
                            label='rain_precip_mm_1hr'):
        """
        Expand weather data row to intervals for a given label
          (default is 'rain_precip_mm_1hr').

        Args:
        - row (pd.Series): A row of the weather dataframe.
        - clip_start (pd.Timestamp): Start timestamp of the audio clip.
        - clip_end (pd.Timestamp): End timestamp of the audio clip.
        - excerpt_length (int, optional): Length of the intervals in seconds.
             Default is 10.
        - label (str, optional): Column name to extract values from.
            Default is 'rain_precip_mm_1hr'.

        Returns:
        - intervals_start (pd.DatetimeIndex): Start timestamps of the intervals.
        - intervals_end (pd.DatetimeIndex): End timestamps of the intervals.
        - values (np.array): Values of the given label for each interval.
        """
        start = max(row['start_date_time'], clip_start)
        end = min(row['end_date_time'], clip_end)

        intervals_start = pd.date_range(
            start=start,
            end=end,
            freq=f"{excerpt_length}s",
            inclusive="left",
        )
        intervals_end = intervals_start + pd.Timedelta(seconds=excerpt_length)

        values = np.full(len(intervals_start), row[label])
        return intervals_start, intervals_end, values

    @staticmethod
    def extract_weather_values_for_clip(
        clip_path,
        weather_df,
        recordings_df,
        excerpt_length=10,
        label='rain_precip_mm_1hr',
    ):
        """
        Extract rain values for each interval of an audio clip.

        Args:
        - clip_path (str): Path of the audio clip.
        - weather_df (pd.DataFrame): Weather dataframe for the clip.
        - recordings_df (pd.DataFrame): Recordings dataframe containing
             audio clip details.

        Returns:
        - torch.Tensor: Rain values for each interval of the audio clip.

        Usage:
        # Build a dictionary: audio clip path -> rain values for each
         interval of the clip

        clip2yvalues_dict = {
            clip_path: extract_weather_values_for_clip(clip_path, weather_df,
                                                    recordings.files)
            for clip_path, weather_df in recording2weather.items()
        }
        """
        clip_start = recordings_df.loc[str(clip_path), 'start_date_time']
        clip_end = recordings_df.loc[str(clip_path), 'end_date_time']

        # Filter weather rows that overlap with the audio clip
        overlapping_mask = (weather_df['end_date_time'] > clip_start) & (
            weather_df['start_date_time'] < clip_end)
        overlapping_weather = weather_df[overlapping_mask]

        clip_rain_values = []
        for _, row in overlapping_weather.iterrows():
            # Reverted unpacking to only keep rain_values
            _, _, rain_values = RecordingsDataset.expand_to_intervals(
                row,
                clip_start,
                clip_end,
                excerpt_length=excerpt_length,
                label=label)
            clip_rain_values.extend(rain_values)
        clip_rain_values = torch.tensor(clip_rain_values, dtype=torch.float32)
        clip_rain_values = RecordingsDataset.apply_threshold2labels(
            clip_rain_values, threshold=0)
        return clip_rain_values

    @staticmethod
    def apply_threshold2labels(labels, threshold):
        """Apply threshold to labels."""
        return (labels > threshold).float()


class TeacherModel():
    """ Teacher model for semi-supervised learning.
    """

    def __init__(self, config=None, device=None):
        self.device = device
        threshold_file = 'real/1esfp153-V1/1esfp153_thresholds.csv'
        if config is None:
            run_id = '1esfp153'
            model_folder = 'runs_models/edansa/augmentV5/checkpoints/run-20231111_161208-1esfp153/'
            config = {
                'run_id':
                    run_id,
                'run_identity':
                    f'nna/megan/{run_id}',
                'model_id':
                    run_id + '-V1',
                'model_path':
                    model_folder + 'best_model_402_val_f1_min=0.8122.pt',
                'config_file':
                    model_folder + '1esfp153_config.json',
            }

        class Args:
            run_identity = config['run_identity']
            model_id = config['run_id'] + '-V1'
            model_path = config['model_path']
            clipping_path = ''
            files_metadata = ''
            output_folder = ''
            gpu = 0
            config_file = config[
                'config_file']  # Set this to the path of your config file if you have one

        args = Args()

        run_dir = '/home/enis/projects/experiments/edansa/runs/augment/'

        # change dir to experiment run directory, so we can import inference
        os.chdir(run_dir)
        # pylint: disable=import-outside-toplevel
        import inference  # type: ignore
        model, config, _, _ = inference.setup(args)
        _ = model.requires_grad_(False)
        self.label_names = [
            config['code2excell_names'][taxo] for taxo in config['target_taxo']
        ]
        if self.device is not None:
            config = self.device
        self.to_tensor = edansa.modelarchs.WaveToTensor(
            config['max_mel_len'],
            config['sampling_rate'],
            device=config['device'],
            feature_method=config['arch'].get('feature_method', 'logmel'))
        self.model = model
        self.config = config
        self.thresholds = self.load_thresholds(threshold_file)

    def load_thresholds(self, threshold_file_path):
        # Load the CSV file into a pandas DataFrame
        df = pd.read_csv(threshold_file_path)

        # Set 'label_name' as the index and convert to dictionary
        result_dict = df.set_index('label_name').to_dict(orient='index')
        thresholds = [
            result_dict[label_name]['threshold']
            for label_name in self.label_names
        ]
        return np.array(thresholds)

    def run_inference(self, samples):
        if len(samples) == 0:
            # raise a specific error
            raise Exception('samples is empty')
        # inputs = torch.randn(1, 1, 128, 128).float().to(config['device'])
        # expected_size = self.config['sampling_rate'] * self.config[
        # 'excerpt_length']
        # sound_array = sound_array.reshape(-1, expected_size)
        preds = []
        for inputs in samples:
            inputs = inputs.float().to(self.config['device'])
            inputs = self.to_tensor((inputs, None))[0]
            inputs = inputs.reshape((1, *inputs.shape))
            output = self.model(inputs).to('cpu').data.numpy()
            preds.append(output)
        preds = np.concatenate(preds)
        return preds


class RecordingsDatasetSSL(TorchDataset):
    """Dataset for loading audio files and their corresponding labels."""

    def __init__(
        self,
        recordings: Union[str, AudioRecordingFiles],
        in_memory_sample_count: int,
        # weather_data: Union[str, pd.DataFrame],
        regloc_list: List[Tuple[str, str]],
        excerpt_len=10,
        resample_rate=48000,
        mono=True,
        channels_first=True,
    ):
        self.regloc_list = regloc_list
        self.recordings = self._load_recordings(recordings)
        self.iterator = None  # iter(self.recordings.files.index)
        self.in_memory_sample_count = in_memory_sample_count
        # self.weather_data = self._load_weather_data(weather_data)
        self.excerpt_len = excerpt_len
        self.resample_rate = resample_rate
        self.data = []
        self.current_sample = 0
        self.teacher_model = TeacherModel()
        self.mono = mono
        self.channels_first = channels_first

        # self.recording2weather = self.matchrecordings2weather(
        # recordings=self.recordings, weather=self.weather_data)
        # self.remove_recordings_without_weather()

        # assert len(self.recording2weather) > 0, 'no matching weather data'
        # self.recording2weather = shuffle_dict(self.recording2weather)
        # self.recording2weather_iter = iter(self.recording2weather.items())
        if self.current_sample >= len(self.data):
            self._load_data()

    def get_next_recording(self,):

        def set_iterator():
            recordings_index = list(self.recordings.files.index)
            random.shuffle(recordings_index)
            return iter(recordings_index)

        if self.iterator is None:
            self.iterator = set_iterator()
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = set_iterator()
            return next(self.iterator)

    def _load_recordings(self, recordings):
        if isinstance(recordings, str):
            # do not load if recordings empty string
            recordings = AudioRecordingFiles(recordings, version='v2')
        elif isinstance(recordings, AudioRecordingFiles):
            pass
        else:
            raise Exception('recordings should be str or AudioRecordingFiles')
        if self.regloc_list:
            recordings.filter_by_regloc(regloc_list=self.regloc_list,
                                        inplace=True)
        assert len(recordings) > 0, 'no recordings found'
        return recordings
        # Define a helper function for the task that should be parallelized
    def load_recordingw_labels(self, recording_path):
        audio = Audio(recording_path)
        try:
            audio.load_data(
                store=True,
                resample_rate=self.resample_rate,
                backend=None,
                mono=self.mono,
                normalize=True,
                channels_first=self.channels_first,
            )
        except Exception as e:
            print(f"Exception occurred: {type(e).__name__}, {str(e)}")
            traceback.print_exc()
            return []

        samples = audio.divide_long_sample(excerpt_len=self.excerpt_len)
        if len(samples) == 0:
            return []
        y_values = self.teacher_model.run_inference(samples)
        y_values = 1 / (1 + np.exp(-y_values))
        y_values = (y_values > self.teacher_model.thresholds).astype(int)
        y_values = torch.from_numpy(y_values).float()

        # y_values = self.extract_weather_values_for_clip(
        # recording_path, recs_weather, self.recordings.files)
        return list(zip(samples, y_values))

    def _load_data(self, filecount2load_inparallel=10):
        """Load the next batch of data into memory."""

        # Initialize empty data list
        self.data = []
        samples_loaded = 0

        while samples_loaded < self.in_memory_sample_count:

            recording_paths = [
                self.get_next_recording()
                for _ in range(filecount2load_inparallel)
            ]

            # Use a ThreadPool to parallelize the loading process
            with ThreadPoolExecutor() as executor:
                results = list(
                    executor.map(self.load_recordingw_labels, recording_paths))

            # Flatten the list of results and append to self.data
            self.data.extend([item for sublist in results for item in sublist])

            samples_loaded += sum(len(sublist) for sublist in results)

        random.shuffle(self.data)

    def __len__(self):
        # return len(self.data)
        return self.in_memory_sample_count

    def __getitem__(self, idx):
        del idx
        if self.current_sample >= len(self.data):
            self._load_data()
            self.current_sample = 0
        sample, label = self.data[self.current_sample]
        self.current_sample += 1

        return sample, label

    @staticmethod
    def apply_threshold2labels(labels, threshold):
        """Apply threshold to labels."""
        return (labels > threshold).float()
