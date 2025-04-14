'''
Utility functions for reading and writing files, mainly predictions and 
embeddings.
'''
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Union, Optional
import datetime
import glob
import logging

# in labelinapp, at samplingutils.py, also change there
TIMESTAMP_ARRAY_KEY = 'timestamp'
TIMESTAMP_INFILE_FORMAT = '%Y-%m-%d_%H:%M:%S'
TIMESTAMP_FILENAME_FORMAT = '%Y-%m-%d_%H-%M-%S'
TIMESTAMP_DATASET_FORMAT = '%Y-%m-%dT%H:%M:%S.%f'
LENGTH_DATASET_FORMAT = '%H:%M:%S.%f'

FILE_PATH_COL = 'file_path'


class InferenceResultsFinal():
    """A container for model predictions on all recordings."""

    def __init__(
            self,
            output_folder=None,
            threshold_info=None,
            excerpt_len=10,
            load=True,
            io=None,
            result_type='binary',  # or 'prob'
    ):
        """A container for model predictions on all recordings."""

        self.output_folder = output_folder
        self.threshold_info = threshold_info
        self.result_type = result_type

        if io is None and output_folder is not None:
            self.io = IO(
                excerpt_len=excerpt_len,
                output_folder=output_folder,
            )
        else:
            self.io = io

        if load:
            if self.io is None:
                raise ValueError('Must provide io or output_folder')
            self.pred_files_pattern = self.io.get_glob_pattern_for_all_csvs()
            self.load()

    def __str__(self) -> str:
        return f'InferenceResultsFinal' + \
            f' with {len(self.all_files)} files'

    def load(self, threshold_info=None):
        self.all_files = glob.glob(self.pred_files_pattern)

        threshold_info = threshold_info or self.threshold_info

        all_results = []
        if not self.all_files:
            raise ValueError(f'No files found at {self.pred_files_pattern},' +
                             ' check if the csv gathering is also run')

        for result_file in self.all_files:
            # get the location, region, and year from the file path
            #  using parse_csv_output_path
            (freq_str, region, location, year,
             orig_file_stem) = self.io.parse_csv_output_path(  # type: ignore
                 result_file)
            del freq_str, orig_file_stem

            results = pd.read_csv(result_file)
            results['location'] = location
            results['region'] = region
            results['year'] = year
            # turn timestamp into datetime
            results[TIMESTAMP_ARRAY_KEY] = pd.to_datetime(
                results[TIMESTAMP_ARRAY_KEY], format=TIMESTAMP_FILENAME_FORMAT)
            all_results.append(results)
        results = pd.concat(all_results)
        self.results = results
        if self.result_type == 'binary':
            self.apply_thresholds()

    def apply_thresholds(self, results=None, threshold_info=None):
        results = results if results is not None else self.results
        threshold_info = threshold_info or self.threshold_info
        assert threshold_info is not None, 'Must provide thresholds'

        def threshold_func(threshold):
            return lambda x: 1 if x >= threshold else 0

        if isinstance(threshold_info, dict):
            for label_name, threshold in threshold_info.items():
                results[label_name] = results[label_name].apply(
                    threshold_func(threshold))
        elif isinstance(threshold_info, pd.DataFrame):
            for _, row in threshold_info.iterrows():
                label_name = row['label_name']
                threshold = row['threshold']
                results[label_name] = results[label_name].apply(
                    threshold_func(threshold))
        else:
            raise ValueError('threshold_info must be dict or DataFrame')
        return results


class AudioRecordingFiles():
    """A container for audio recording files.

    Wrapper around a list of audio recording files. Provides methods for
    accessing the files and their metadata.

    Currently uses dataframe as a backend, but this may change in the future.
    """

    def __init__(
        self,
        dataframe: Union[pd.DataFrame, Path, str],
        version='v2',
    ):
        self.version = version
        if isinstance(dataframe, pd.DataFrame):
            self.files = dataframe
            self.files.set_index(FILE_PATH_COL, inplace=True, drop=False)
        elif isinstance(dataframe, (Path, str)):
            dataframe_path = Path(dataframe)
            if dataframe_path.suffix == '.pkl':
                dataframe = pd.read_pickle(str(dataframe_path))
                dataframe.set_index(  # type: ignore
                    FILE_PATH_COL, inplace=True, drop=False)
            elif dataframe_path.suffix == '.csv':
                dataframe = pd.read_csv(str(dataframe_path),
                                        index_col=FILE_PATH_COL,
                                        dtype={
                                            FILE_PATH_COL: str,
                                            'region': str,
                                            'location': str
                                        })
            else:
                raise ValueError('unknown file type for current_metadata_file')
        else:
            raise ValueError('dataframe must be a DataFrame or Path-str')
        assert isinstance(dataframe, pd.DataFrame)
        self.files = dataframe
        self.fix_location_key()
        self.update_col_names()
        self.strptimecolumns()

    def strptimecolumns(self,):
        # Convert columns to datetime only if they exist
        if 'start_date_time' in self.files.columns:
            try:
                self.files['start_date_time'] = pd.to_datetime(
                    self.files['start_date_time'],
                    format=TIMESTAMP_DATASET_FORMAT,
                    errors='coerce'  # Handle potential parsing errors gracefully
                )
                # Calculate year only if start_date_time was successfully converted
                if pd.api.types.is_datetime64_any_dtype(
                        self.files['start_date_time']):
                    self.files['year'] = self.files['start_date_time'].dt.year
                else:
                    # Handle case where conversion resulted in NaT
                    self.files['year'] = pd.NA
            except Exception as e:
                logging.warning(f"Could not parse 'start_date_time': {e}")
                self.files[
                    'year'] = pd.NA  # Ensure year column exists but has NA

        if 'end_date_time' in self.files.columns:
            try:
                self.files['end_date_time'] = pd.to_datetime(
                    self.files['end_date_time'],
                    format=TIMESTAMP_DATASET_FORMAT,
                    errors='coerce'  # Handle potential parsing errors gracefully
                )
            except Exception as e:
                logging.warning(f"Could not parse 'end_date_time': {e}")

        # Ensure duration_sec is float, handle potential errors
        if 'duration_sec' in self.files.columns:
            self.files['duration_sec'] = pd.to_numeric(
                self.files['duration_sec'], errors='coerce')
            self.files['duration_sec'] = self.files['duration_sec'].astype(
                float)
        else:
            # Attempt to calculate from start/end if they exist and are valid datetimes
            if ('start_date_time' in self.files.columns and
                    'end_date_time' in self.files.columns and
                    pd.api.types.is_datetime64_any_dtype(
                        self.files['start_date_time']) and
                    pd.api.types.is_datetime64_any_dtype(
                        self.files['end_date_time'])):
                # Calculate duration if possible, coercing errors to NaT
                duration = pd.to_timedelta(self.files['end_date_time'] -
                                           self.files['start_date_time'],
                                           errors='coerce')
                # Convert valid timedeltas to seconds (float), leave NaT as NaN
                self.files['duration_sec'] = duration.dt.total_seconds().astype(
                    float)
                logging.info("Calculated 'duration_sec' from start/end times.")
            else:
                logging.warning(
                    "'duration_sec' column missing and could not be calculated."
                )

    def update_col_names(self, version=None):
        if version is None:
            version = self.version
        if version.lower() == 'v1':
            return True
        elif version.lower() == 'v2':
            expected_cols = [
                'region',
                'location',
                'recorder_id',
                'start_date_time',
                'end_date_time',
                'length',
                'duration_sec',
                'file_path',
            ]
            if all([col in self.files.columns for col in expected_cols]):
                return True
            # we want to keep
            ## region, location, clip_path, recorder_id, start_date_time,
            #  end_date_time, length
            # delete columns that are not in the v1 from dataframe
            self.files.drop(columns=[
                'site_id',
                'site_name',
                'hour_min_sec',
                'year',
                'month',
                'day',
                'locationId',
            ],
                            inplace=True,
                            errors='ignore')
            #rename column recorderId to recorder_id
            # ignores if the column does not exist
            self.files.rename(columns={
                'recorderId': 'recorder_id',
                'timestamp': 'start_date_time',
                'timestampEnd': 'end_date_time',
                'durationSec': 'duration_sec',
            },
                              inplace=True)
            if 'duration_sec' not in self.files.columns:
                self.files['duration_sec'] = self.files['length']
            if 'length' not in self.files.columns:
                self.files['length'] = self.files['duration_sec']
            # make it string,but not lowercase
            self.files['location'] = self.files['location'].astype(str)
            self.files['region'] = self.files['region'].astype(str)
            #sort columns
            #turn index into a column
            self.files['file_path'] = self.files.index
            self.files.index = self.files.index.astype(str)
            # self.files.reset_index(drop=True, inplace=True)
            cols = self.files.columns.tolist()
            expected_cols.extend([x for x in cols if x not in expected_cols])
            # Select only the expected columns that are actually present in the dataframe
            present_expected_cols = [
                col for col in expected_cols if col in self.files.columns
            ]
            # Ensure essential columns (like file_path if used as index or column) are kept if present
            # (Assuming FILE_PATH_COL is defined elsewhere and is 'file_path')
            if FILE_PATH_COL not in present_expected_cols and FILE_PATH_COL in self.files.columns:
                present_expected_cols.insert(0, FILE_PATH_COL)

            # Reindex using only the present columns from the expected list
            self.files = self.files[present_expected_cols]
            return True
        else:
            raise ValueError('version must be v1 or v2')

    def fix_location_key(self,):
        """turn locationId to location
        """
        if 'locationId' in self.files.columns:
            self.files['location'] = self.files['locationId']
        # self.files = self.files.drop(columns=['locationId'])
    def set_year_from_timestamp(self,):
        self.files['year'] = self.files['start_date_time'].dt.year

    def __len__(self):
        if isinstance(self.files, pd.DataFrame):
            return len(self.files.index)
        else:
            return len(self.files)

    def __getitem__(self, key):
        if isinstance(key, int):
            try:
                return self.files.iloc[key]  # type: ignore
            except IndexError:
                print(f'Index {key} out of range')
                return None
        if isinstance(key, Path):
            key = str(key)
        try:
            return self.files.loc[key]  # type: ignore
        except KeyError:
            # print(f'Warning: Key of {key} not found in AudioRecordingFiles')
            return None

    def filter(self,
               region=None,
               location=None,
               year=None,
               duration_sec=None,
               inplace=False):
        """Filter the files by region, location, and year.
        """
        df = self.files
        filter_list = []
        if region is not None:
            filter_list.append(df['region'] == region)
        if location is not None:
            filter_list.append(df['location'] == location)
        if year is not None:
            filter_list.append(df['year'] == year)
        if duration_sec is not None:
            filter_list.append(df['duration_sec'] >= duration_sec)
        if len(filter_list) == 0:
            return df if inplace else df.copy()
        else:
            for i in filter_list[1:]:
                filter_list[0] = filter_list[0] & i

        if inplace:
            self.files = df[filter_list[0]].copy()
            return self.files
        else:
            return df[filter_list[0]].copy()

    def filter_by_regloc(self, regloc_list, inplace=False):
        """Filter the files by region and location.
        """
        df = self.files
        # regloc_list = [(x[0].lower(), x[1].lower()) for x in regloc_list]
        filtered_df = df[df.set_index(['region',
                                       'location']).index.isin(regloc_list)]
        # filtered_df.reset_index(drop=True, inplace=True)
        if inplace:
            self.files = filtered_df
            return self.files
        else:
            return filtered_df

    def years_set(self, region=None, location=None):
        """Return a list of years that have recordings.
        """
        df = self.filter(region=region, location=location)
        return set(df['start_date_time'].dt.year.unique())

    def groupby_regloc(self):
        """Return a list of regions and locations that have recordings.
        """
        return self.files.groupby(['region', 'location'])

    def get_time_period(self,):
        """Return the earliest and latest recording times.
        """
        df = self.files
        return df['start_date_time'].min(), df['end_date_time'].max()


class IO():
    """
    A class for managing input and output paths.

        Args:
            excerpt_len (int): excerpt length in seconds
            output_folder (str): output folder, for the prediction files

    """

    def __init__(
        self,
        excerpt_len: int = 10,
        output_folder: Optional[str] = None,
    ):
        """Initializes the IO handler.

        Args:
            excerpt_len: Length of audio excerpts in seconds.
            output_folder: Base directory for saving outputs. If None, defaults
                           to a standard scratch path.
        """
        self.excerpt_len = excerpt_len

        # Determine and create the base output path
        self.output_folder = self._resolve_output_path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)

    def _resolve_output_path(self, output_folder: Optional[str]) -> Path:
        if output_folder is None:
            # Default to creating an 'outputs' directory in the current working directory
            default_path = Path("outputs")
            logging.info(
                f"Output folder not specified, using default: {default_path.resolve()}"
            )
            return default_path
            # return Path('/scratch/enis/predictions') # OLD DEFAULT
        else:
            return Path(output_folder)

    @property
    def raw_freq_str(self):
        return 'raw'

    @property
    def csv_freq_str(self,):
        return f'{self.excerpt_len}s_csv'

    @property
    def pred_output_type_str(self):
        return 'pred'

    # function to add output path to relative path
    def add_parent_output_folder(self, path: Union[Path, str]):
        return self.output_folder / path

    def per_file_pred_output_path(
        self,
        region,
        location_id,
        year: Union[int, str],
        original_file_stem,
    ):
        """
        opionined prediction output file name for a given sigle input file

        f'{raw}/{region}/{location_id}/{year}/{original_file.stem}-{output_type}'

        """  # pylint: disable=C0301:line-too-long

        return self.pred_output_file_name(
            self.raw_freq_str,
            region,
            location_id,
            year,
            original_file_stem,
        )

    def excerpt_len_pred_output_path(
        self,
        region: str,
        location_id: str,
        year: Union[int, str],
        timestamp: Union[pd.Timestamp, datetime.datetime, str],
        excerpt_len=None,
    ):
        """
        name of the file for merged predictions from same location and year
        this file is generated in post-process stage by concat. raw results

        path: {excerpt_length}/{region}/{location_id}/{year}/`
        filename: {timestamp}-{output_type}.npz
                timestamp is in format '%Y-%m-%d_%H-%M-%S',
                it is the start time of the earliest recording

        """
        excerpt_len = self.excerpt_len if excerpt_len is None else excerpt_len

        freq_str = f'{excerpt_len}s'
        return self.aggregated_pred_output_path(region, location_id, year,
                                                timestamp, freq_str)

    def aggregated_pred_output_path(
        self,
        region,
        location_id,
        year: Union[int, str],
        timestamp,
        freq_str,
    ):
        """
        name of the file for aggregated predictions from same location and year
        freq_str is the frequency of the aggregated predictions
        filename is in format '{timestamp}-{output_type}.npz'

        f'{freq_str}/{region}/{location_id}/{year}/{timestamp}-{output_type_str}.npz'
        """  # pylint: disable=C0301:line-too-long

        if freq_str.lower() == self.raw_freq_str:
            raise ValueError(f'freq_str cannot be {self.raw_freq_str}')
        if not isinstance(timestamp, str):
            timestamp = timestamp.strftime(TIMESTAMP_FILENAME_FORMAT)
        return self.pred_output_file_name(
            freq_str,
            region,
            location_id,
            year,
            timestamp,
        )

    def pred_output_file_name(
        self,
        freq_str,
        region,
        location_id,
        year: Union[int, str],
        output_stem_root,
    ):
        """ 
        name of the file for aggregated predictions from same location and year
        freq_str is the frequency of the aggregated predictions
        filename is in format '{output_stem_root}-{self.pred_output_type_str}.npz'
        
        f'{freq_str}/{region}/{location_id}/{year}/{output_stem_root}-{self.pred_output_type_str}.npz'
        """# pylint: disable=C0301:line-too-long
        output_type_str = self.pred_output_type_str
        file_ext = '.npz'

        return self.output_file_path(freq_str, region, location_id, year,
                                     output_stem_root, output_type_str,
                                     file_ext)

    def output_file_path(
        self,
        freq_str,
        region,
        location_id,
        year: Union[int, str],
        output_stem,
        output_type_str,
        file_ext='',
    ):
        """
        generic output file path and name
        
        Args:
            freq_str (str): pandas dataframe frequency string
            region (str): region
            location_id (str): location id
            year (str): year
            output_stem_root (str): output file stem root
        """

        folder_name = self.common_folder_hierarchy(freq_str, region,
                                                   location_id, year)
        file_name = f'{output_stem}_{output_type_str}{file_ext}'
        return Path(folder_name) / file_name

    def common_folder_hierarchy(
        self,
        freq_str,
        region,
        location_id,
        year: Union[int, str],
    ):
        """
        common middle folder hierarchy for all output files

        {freq_str}/{region}/{location_id}/{year}/
        """
        return f'{freq_str}/{region}/{location_id}/{year}'

    def get_glob_pattern_for_raw_preds(
        self,
        region,
        location_id,
        year: Union[int, str],
    ):
        """
        glob pattern for raw predictions
        """
        folder = self.add_parent_output_folder(
            self.common_folder_hierarchy(self.raw_freq_str, region, location_id,
                                         year))
        glob_str = str(folder) + f'/*_{self.pred_output_type_str}.npz'
        return glob_str

    def save_per_file_pred2disk(
        self,
        file_row,
        preds,
        target_taxos,
    ):
        """
        saves prediction for a single file to disk
        """

        nparrays_dict = {}
        for i, target_taxo in enumerate(target_taxos):
            nparrays_dict[target_taxo] = preds[:, i]
        output_file_path = self.per_file_pred_output_path(
            file_row.region, file_row.location, file_row.year,
            Path(file_row.name).stem)
        output_file_path = self.add_parent_output_folder(output_file_path)
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(output_file_path), **nparrays_dict)

    def save_concat_preds2disk(
        self,
        region: str,
        location: str,
        year: Union[int, str],
        preds: Dict[str, np.ndarray],
        start_timestamp: Union[pd.Timestamp, datetime.datetime, str],
        excerpt_len: int,
    ):
        """
        saves aggregated predictions to disk, used in post processing
        """

        filename = self.excerpt_len_pred_output_path(
            region,
            location,
            year,
            timestamp=start_timestamp,
            excerpt_len=excerpt_len,
        )
        output_path = self.add_parent_output_folder(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(str(output_path), **preds)

    def is_realtive_path_exist(self, relative_file_path):
        return self.add_parent_output_folder(relative_file_path).exists()

    def is_per_file_pred_exist(self, file_row):
        """_summary_

        Args:
            file_row (pd.Series): dataframe row with file information

        Returns:
            Bool: whether prediction exists or not
        """

        output_file_path = self.per_file_pred_output_path(
            file_row.region, file_row.location, file_row.year,
            Path(file_row.name).stem)
        return self.is_realtive_path_exist(output_file_path)

    def get_expected_output_path(
        self,
        result_type: str,
        audio_file_path: Path,
        input_data_root: Path,
    ) -> Path:
        """Calculates the expected output path for a given audio file and result type.

        Mirrors the path calculation logic from `save_results_per_file`.

        Args:
            result_type: Specifies whether 'predictions' or 'embeddings' are expected.
            audio_file_path: The absolute path to the original input audio file.
            input_data_root: The absolute path to the root directory from which the relative
                             path of the audio file should be calculated.

        Returns:
            The calculated absolute Path object for the expected output file.

        Raises:
            ValueError: If the relative path cannot be determined or result_type is unknown.
        """
        try:
            # 1. Calculate relative path
            relative_path = audio_file_path.relative_to(input_data_root)

            # 2. Determine the correct suffix based on result_type
            if result_type == 'predictions':
                output_suffix = '.csv'
            elif result_type == 'embeddings':
                output_suffix = '.npz'
            else:
                raise ValueError(f"Unknown result_type: {result_type}")

            # 3. Construct the full output path
            output_path = Path(
                self.output_folder) / relative_path.with_suffix(output_suffix)
            return output_path

        except ValueError as e:
            # Handle potential error in relative_to()
            logging.error(
                f"Path calculation error for {audio_file_path} relative to {input_data_root}: {e}"
            )
            raise  # Re-raise the ValueError

    def save_aggregated_as_csv(
        self,
        merged_df,
        region,
        location,
        year: Union[int, str],
        pred_file_stem,
        excerpt_len=None,
    ):
        """Save the aggregated dataframe to a CSV file.

        Args:
            merged_df: DataFrame containing aggregated predictions and possibly clipping.
            region: Region identifier.
            location: Location identifier.
            year: Year identifier.
            pred_file_stem: The stem for the output filename (usually derived from first timestamp).
            excerpt_len: Excerpt length in seconds (used to determine frequency string).
        """
        excerpt_len = self.excerpt_len if excerpt_len is None else excerpt_len

        freq_str = f'{excerpt_len}S'  # Frequency string based on excerpt length

        # Determine output path using existing helper method
        output_path = self.output_file_path(
            freq_str=freq_str,
            region=region,
            location_id=location,
            year=year,
            output_stem=pred_file_stem,
            output_type_str='predictions',
            file_ext='.csv',
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        merged_df.to_csv(str(output_path),
                         index=True)  # Ensure index (timestamp) is saved

    def save_results_per_file(
        self,
        results: Union[pd.DataFrame,
                       Dict[str, np.ndarray]],  # Accept Dict for embeddings
        result_type: str,  # 'predictions' or 'embeddings'
        audio_file_path: Path,
        input_data_root: Path,
    ):
        """Saves inference results (predictions CSV or embeddings NPZ) for a single audio file,
           preserving the relative directory structure.

        Args:
            results: The data to save (DataFrame for predictions, Dict for embeddings).
            result_type: Specifies whether 'predictions' or 'embeddings' are being saved.
            audio_file_path: The absolute path to the original input audio file.
            input_data_root: The absolute path to the root directory from which the relative
                             path of the audio file should be calculated.
        """
        output_path = None  # Initialize for logging in case of early error
        try:
            # 1. Calculate relative path
            relative_path = audio_file_path.relative_to(input_data_root)

            # 2. Construct the final output path including the relative structure
            # The structure is <output_folder>/<relative_audio_path>
            # Determine the correct suffix based on result_type
            if result_type == 'predictions':
                output_suffix = '.csv'
                data_to_save = results
                if not isinstance(data_to_save, pd.DataFrame):
                    raise TypeError(
                        f"Expected pandas DataFrame for predictions, got {type(data_to_save)}"
                    )
            elif result_type == 'embeddings':
                output_suffix = '.npz'
                data_to_save = results
                if not isinstance(data_to_save, dict):
                    raise TypeError(
                        f"Expected dict for embeddings, got {type(data_to_save)}"
                    )
                if 'embeds' not in data_to_save or not isinstance(
                        data_to_save['embeds'], np.ndarray):
                    raise ValueError(
                        "Embeddings dict must contain 'embeds' key with a numpy array."
                    )
            else:
                raise ValueError(f"Unknown result_type: {result_type}")

            # Construct the full output path
            output_path = Path(
                self.output_folder) / relative_path.with_suffix(output_suffix)

            # 4. Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # 5. Save the file
            if result_type == 'predictions':
                # Make sure index (timestamps) is saved
                results_df = data_to_save
                results_df.index.name = TIMESTAMP_ARRAY_KEY

                # Format the timestamp index before saving ONLY if it's datetime
                if pd.api.types.is_datetime64_any_dtype(results_df.index):
                    results_df.index = results_df.index.strftime(
                        TIMESTAMP_INFILE_FORMAT)
                # Otherwise, the index (float seconds) will be saved as is.

                # Save predictions to CSV
                results_df.to_csv(output_path)
                logging.info(f"Saved predictions to {output_path}")

            elif result_type == 'embeddings':
                # Save only the embeddings array under the key 'embeddings'
                save_dict = {'embeddings': data_to_save['embeds']}
                np.savez_compressed(output_path, **save_dict)
                logging.debug(f"Saved embeddings to: {output_path}")

        except ValueError as e:
            # Handle potential error in relative_to() or unknown result_type
            logging.error(
                f"Path or Value error for {audio_file_path} relative to {input_data_root}: {e}"
            )
            raise  # Re-raise or handle as appropriate
        except Exception as e:
            save_path_str = str(
                output_path) if output_path else "[unknown path]"
            logging.error(
                f"Failed to save results for {audio_file_path.name} to {save_path_str}: {e}"
            )
            raise  # Re-raise exception

    def get_glob_pattern_for_all_csvs(self):
        """
        glob pattern for csv predictions

        {parent_output_folder}/{freq_str}/*/*/*/*_{self.pred_output_type_str}.csv
        """ # pylint: disable=C0301:line-too-long
        folder = self.add_parent_output_folder(
            self.common_folder_hierarchy(
                self.csv_freq_str,
                'notavailable',
                'notavailable',
                'notavailable',
            ))
        folder = folder.parent.parent.parent
        glob_str = folder / '*' / '*' / '*' / '*.csv'
        return str(glob_str)

    def parse_csv_output_path(self, output_file_path):
        """
        parse csv output path into its components
        output_file_path = {freq_str}_csv/{region}/{location_id}/{year}/orig_file_stem.csv
        (Assuming it's relative to output_folder)
        """ # pylint: disable=C0301:line-too-long
        output_file_path = Path(output_file_path)
        freq_str = output_file_path.parts[-5].split('_')[0]
        region = output_file_path.parts[-4]
        location_id = output_file_path.parts[-3]
        year = output_file_path.parts[-2]
        orig_file_stem = output_file_path.stem
        return freq_str, region, location_id, year, orig_file_stem
