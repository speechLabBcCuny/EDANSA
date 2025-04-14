import edansa.io
from pathlib import Path
import datetime
import pandas as pd
import numpy as np
import shutil
import pytest


def test_per_file_pred_output_path(tmp_path):
    io = edansa.io.IO(excerpt_len=10, output_folder=str(tmp_path))
    exp_output = Path('raw/region/location_id/year/original_file_stem_pred.npz')
    output = io.per_file_pred_output_path('region', 'location_id', 'year',
                                          'original_file_stem')
    assert exp_output == output


def test_excerpt_len_pred_output_path(tmp_path):
    excerpt_len = 10
    io = edansa.io.IO(excerpt_len=excerpt_len, output_folder=str(tmp_path))
    timestamp = datetime.datetime(2019, 1, 1, 0, 0, 0)
    exp_output = Path(
        '10s/region/location_id/year/2019-01-01_00-00-00_pred.npz')
    output = io.excerpt_len_pred_output_path('region', 'location_id', 'year',
                                             timestamp, excerpt_len)
    output2 = io.excerpt_len_pred_output_path(
        'region',
        'location_id',
        'year',
        timestamp,
    )

    assert output == exp_output
    assert output2 == exp_output


def test_pred_output_file_name(tmp_path):
    io = edansa.io.IO(excerpt_len=10, output_folder=str(tmp_path))
    exp_output = Path(
        'freq_str/region/location_id/year/output_stem_root_pred.npz')
    output = io.pred_output_file_name('freq_str', 'region', 'location_id',
                                      'year', 'output_stem_root')
    assert exp_output == output


def test_output_file_path(tmp_path):
    io = edansa.io.IO(excerpt_len=10, output_folder=str(tmp_path))
    exp_output = Path(
        'freq_str/region/location_id/year/output-stem-root_output-type-str.xyz')
    output = io.output_file_path(
        'freq_str',
        'region',
        'location_id',
        'year',
        'output-stem-root',
        'output-type-str',
        file_ext='.xyz',
    )
    assert exp_output == output


def test_common_folder_hierarchy(tmp_path):
    io = edansa.io.IO(excerpt_len=10, output_folder=str(tmp_path))
    exp_output = ('freq_str/region/location_id/year')
    output = io.common_folder_hierarchy('freq_str', 'region', 'location_id',
                                        'year')
    assert exp_output == output


def test_get_glob_pattern_for_raw_preds(tmp_path):
    io = edansa.io.IO(excerpt_len=10, output_folder=str(tmp_path))
    exp_output = (str(tmp_path / 'raw' / 'dempster' / '10' / '2021' /
                      '*_pred.npz'))
    pattern = io.get_glob_pattern_for_raw_preds('dempster', '10', '2021')
    assert pattern == exp_output


def test_save_concat_preds2disk(tmp_path):
    io = edansa.io.IO(excerpt_len=10, output_folder=str(tmp_path))
    preds = np.array([[1, 2, 3], [4, 5, 6]])


def test_save_per_file_pred2disk(tmp_path):
    io = edansa.io.IO(excerpt_len=10, output_folder=str(tmp_path))
    file_row, model_id, preds, label_names = pd.Series(
        {
            'region': 'dempster',
            'location': '10',
            'year': '2021'
        },
        name=Path('test.wav')), 'model_id', np.array([[1, 2], [4, 5], [7, 8]
                                                     ]), ['label1', 'label2']
    io.save_per_file_pred2disk(file_row, preds, label_names)

    # check if the file is saved
    # Removed model_id from path
    output_file_rel = Path("raw/dempster/10/2021/test_pred.npz")
    output_file_abs = io.output_folder / output_file_rel  # Construct full path
    container = np.load(output_file_abs)  # Load using absolute path
    # Check if the saved data is correct
    e = {name: container[name] for name in container}
    del container
    assert np.array_equal(e['label1'], preds[:, 0])
    assert np.array_equal(e['label2'], preds[:, 1])


def test_is_realtive_path_exist(tmp_path):
    '''
    test if the file exists
    '''
    io = edansa.io.IO(excerpt_len=10, output_folder=str(tmp_path))
    is_file_exist = io.is_realtive_path_exist('empty.txt')
    assert is_file_exist == False
    # Create the dummy file inside the tmp_path managed by the IO instance
    (tmp_path / 'empty.txt').touch()

    is_file_exist = io.is_realtive_path_exist('empty.txt')
    assert is_file_exist == True
    # No need for shutil.rmtree anymore, tmp_path handles cleanup


def test_is_per_file_pred_exist(tmp_path):
    io = edansa.io.IO(excerpt_len=10, output_folder=str(tmp_path))
    file_row, model_id = pd.Series(
        {
            'region': 'dempster',
            'location': '10',
            'year': '2021'
        },
        name=Path('test-exist.wav')), 'model_id'
    # Construct the expected path *relative* to the IO output folder (tmp_path)
    expected_relative_path = io.per_file_pred_output_path(
        file_row['region'], file_row['location'], file_row['year'],
        file_row.name.stem)
    # Create the dummy file at the correct absolute path within tmp_path
    absolute_path_to_create = tmp_path / expected_relative_path
    absolute_path_to_create.parent.mkdir(parents=True, exist_ok=True)
    absolute_path_to_create.touch()

    is_file_exist = io.is_per_file_pred_exist(file_row)
    assert is_file_exist
    # test not exist
