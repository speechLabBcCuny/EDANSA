'''Tests for dataimport module.

'''

from edansa import dataimport
import pytest
import numpy as np
import torch
import pandas as pd
from edansa.io import AudioRecordingFiles

test_data_data_to_samples = [
    (10, 54, 1, 48000.0, 'random/path/to/audio/file'),
    (10, 5.1, 1, 48000.0, 'random/path/to/audio/file'),
    (10.0, 5.1, 1, 48000.0, 'random/path/to/audio/file'),
    (10.0, 58.1, 1, 48000, 'random/path/to/audio/file'),
    (10.0, 54.1, 1, 48000, 'random/path/to/audio/file'),
    (20.0, 54.1, 1, 48000, 'random/path/to/audio/file'),
    (10.0, 54.1, 2, 48000, 'random/path/to/audio/file'),
    (10.0, 56.1, 2, 48000, 'random/path/to/audio/file'),
]


@pytest.mark.parametrize(
    'excerpt_len,audio_file_len,channel_count,audio_sr,audio_file_path',
    test_data_data_to_samples,
    # indirect=True,
)
def test_data_to_samples(excerpt_len, audio_file_len, channel_count, audio_sr,
                         audio_file_path):
    print(excerpt_len, audio_file_len, channel_count, audio_sr, audio_file_path)
    # excerpt_len,audio_file_len,audio_sr,audio_file_path  = inputs

    sound_ins = dataimport.Audio(audio_file_path, audio_file_len)
    sound_ins.sr = audio_sr
    if channel_count == 1:
        sound_ins.data = np.ones((1, int(audio_file_len * audio_sr)))
    else:
        sound_ins.data = np.ones(
            (channel_count, int(audio_file_len * audio_sr)))

    sound_ins.data_to_samples(excerpt_len=excerpt_len)

    assert isinstance(sound_ins.samples, list)
    if audio_file_len > 10:
        sample_count = audio_file_len // excerpt_len
        trim_point = int(sample_count * excerpt_len * sound_ins.sr)
        if audio_file_len % excerpt_len >= 5:
            # test if after trim_point is all zeros
            assert np.sum(sound_ins.samples[-1][trim_point:]) == 0
            sample_count += 1
        # test that the number of samples is correct
        assert len(sound_ins.samples) == sample_count
    else:
        assert len(sound_ins.samples) == 1
    if channel_count == 1:
        assert np.sum(sound_ins.samples[0][int(audio_file_len *
                                               audio_sr):]) == 0
    else:
        assert np.sum(sound_ins.samples[0][:,
                                           int(audio_file_len *
                                               audio_sr):]) == 0
    assert sound_ins.samples[0].size == audio_sr * excerpt_len * channel_count

    bb = np.array(sound_ins.samples)
    _ = torch.from_numpy(bb)


def test_mono_audio():
    processor = dataimport.Audio('', 0)
    processor.data = np.array([i for i in range(44100 * 10)
                              ])  # 10 seconds of mono audio

    # Create 5-second excerpts
    excerpts = processor.divide_long_sample(excerpt_len=5, sr=44100)

    # Check if we have two excerpts (5 seconds each)
    assert len(excerpts) == 2
    assert len(excerpts[0]) == 44100 * 5
    assert len(excerpts[1]) == 44100 * 5


def test_stereo_audio():
    processor = dataimport.Audio('', 0)
    processor.data = np.array([[i, i + 1] for i in range(44100 * 29)
                              ]).T  # 29 seconds of stereo audio

    # Create 10-second excerpts
    excerpts = processor.divide_long_sample(excerpt_len=10, sr=44100)

    # Check if we have three excerpts (10 seconds each) and
    #  one 9 seconds (to be padded)
    assert len(excerpts) == 3

    # Validate the shape of each excerpt
    assert excerpts[0].shape == (2, 44100 * 10)
    assert excerpts[1].shape == (2, 44100 * 10)
    assert excerpts[2].shape == (2, 44100 * 10
                                )  # This one is padded to 10 seconds

    first_few = np.array([[i, i + 1] for i in range(10)]).T
    assert (first_few == excerpts[0][:, :10]).all()
    assert (excerpts[2][:, -44100 * 1:] == 0
           ).all()  # The last second should be zeros due to padding


def test_short_audio():
    processor = dataimport.Audio('', 0)
    processor.data = np.array([i for i in range(44100 * 6)
                              ])  # 10 seconds of mono audio

    # Create 5-second excerpts
    excerpts = processor.divide_long_sample(excerpt_len=10, sr=44100)
    # Check if we have two excerpts (5 seconds each)
    assert len(excerpts) == 1
    assert len(excerpts[0]) == 44100 * 10


@pytest.fixture
def mock_data():
    """Return mock data for testing purposes."""
    recording2weather = {
        '/path/to/clip1':
            pd.DataFrame({
                'start_date_time': [
                    pd.Timestamp('2019-05-04 00:00:00'),
                    pd.Timestamp('2019-05-04 01:00:00')
                ],
                'end_date_time': [
                    pd.Timestamp('2019-05-04 01:00:00'),
                    pd.Timestamp('2019-05-04 02:00:00')
                ],
                'rain_precip_mm_1hr': [1.0, 0.0]
            }),
    }

    recordings = pd.DataFrame(
        {
            'start_date_time': [pd.Timestamp('2019-05-04 00:40:00')],
            'end_date_time': [pd.Timestamp('2019-05-04 01:30:00')],
        },
        index=['/path/to/clip1'])

    return recording2weather, recordings


def test_expand_to_intervals(mock_data):  # pylint: disable=redefined-outer-name

    recording2weather, _ = mock_data
    weather_df = recording2weather['/path/to/clip1']
    row = weather_df.iloc[0]

    clip_start = pd.Timestamp('2019-05-04 00:00:00')
    clip_end = pd.Timestamp('2019-05-04 01:00:00')

    (intervals_start, intervals_end,
     values) = dataimport.RecordingsDataset.expand_to_intervals(
         row, clip_start, clip_end)

    expected_start = pd.date_range(clip_start,
                                   clip_end,
                                   freq='10s',
                                   inclusive='left')
    expected_end = expected_start + pd.Timedelta(seconds=10)

    assert np.array_equal(intervals_start, expected_start)
    assert np.array_equal(intervals_end, expected_end)
    assert all(v == 1.0 for v in values)


def test_extract_weather_values_for_clip(mock_data):  # pylint: disable=redefined-outer-name
    recording2weather, recordings = mock_data
    clip_path = '/path/to/clip1'
    weather_df = recording2weather[clip_path]

    result = dataimport.RecordingsDataset.extract_weather_values_for_clip(
        clip_path, weather_df, recordings).numpy()

    # 120 values from first hour (since clip starts at 00:40)
    #  + 180 values from the second hour
    # also applied threshold at 0 mm/h
    expected_values = np.array([1.0] * 120 + [0] * 180)
    print(result.shape)
    print(expected_values.shape)
    # assert np.array_equal(result, expected_values)
    # Use allclose for float comparison
    assert np.allclose(result, expected_values)
