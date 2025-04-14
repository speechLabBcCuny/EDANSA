"""Process excell from megan and taxonomy.yaml to create dataset.

Exported from prepare_dataset notebook.
"""

# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from pathlib import Path
import csv
from pprint import pprint

import edansa
import edansa.dataimport


def delete_samples_by_length_limit(audio_dataset, sample_length_limit):
    """find samples that are not long enough and delete samples from dataset

    """
    sample_not_long_enough = []
    for k, v in audio_dataset.items():
        if v.length < sample_length_limit:
            sample_not_long_enough.append(k)

    #  DELETE samples with not enough data
    for k in sample_not_long_enough:
        del audio_dataset[k]
    if len(sample_not_long_enough) > 0:
        pprint(
            'WARNING: ' +
            f'-> {len(sample_not_long_enough)} number of samples are deleted ' +
            'because their length is not long enough.')

    return sample_not_long_enough


#%%
def load_csv(csv_path):
    with open(csv_path) as csvfile:
        reader = csv.DictReader(csvfile)
        reader = list(reader)
        reader_strip = []
        for row in reader:
            row = {r: row[r].strip() for r in row}
            reader_strip.append(row)
        reader = reader_strip.copy()
    return reader


def run(
    dataset_csv_path,
    taxonomy_file_path,
    ignore_files,
    excerpt_length,
    sample_length_limit,
    excell_names2code=None,
    dataset_name_v='',
    dataset_cache_folder='',
    dataset_folder=None,
    version='V2',
    load_clipping=True,
    target_taxo=None,
):

    megan_data_sheet = load_csv(dataset_csv_path)
    audio_dataset = edansa.dataimport.Dataset(
        megan_data_sheet,
        dataset_name_v=dataset_name_v,
        excerpt_len=excerpt_length,
        dataset_cache_folder=dataset_cache_folder,
        dataset_folder=dataset_folder,
        excell_names2code=excell_names2code,
        taxonomy_file_path=taxonomy_file_path,
        target_taxo=target_taxo,
    )

    sample_not_long_enough = delete_samples_by_length_limit(
        audio_dataset, sample_length_limit)
    if load_clipping:
        audio_dataset.update_samples_w_clipping_info()
    deleted_files = (sample_not_long_enough)
    return audio_dataset, deleted_files
