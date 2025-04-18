# EDANSA: Pre-trained Bioacoustic Models

[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://speechlabbccuny.github.io/EDANSA/) <!-- Placeholder for actual URL -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/speechLabBcCuny/EDANSA/blob/main/notebooks/edansa_inference_example.ipynb)

This repository provides pre-trained models for detecting bioacoustic events and specific bird species in Passive Acoustic Monitoring (PAM) data, developed as part of the EDANSA project.

## Available Models

Two primary pre-trained models are available in the `assets/` directory:

1.  **General Bioacoustics Model (`31m2plxv-V1`):** Detects broader sound categories (Biophony, Anthropophony, Birds, Insects, Silence, etc.). See [Model Card](./docs/using_pretrained_model/model_card_general_31m2plxv-V1.md).
2.  **Bird Species Model (`ppq7zxqq`):** Detects specific North American arctic bird species (Lapland Longspur, White-crowned Sparrow, etc.). See [Model Card](./docs/using_pretrained_model/model_card_bird_species_ppq7zxqq.md).

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/speechLabBcCuny/EDANSA.git
    cd EDANSA
    ```
2.  **Create Conda Environment (Recommended):** Use the provided environment file. You must specify an environment name (e.g., `edansa`).
    ```bash
    conda env create -f environment.yml -n <your_env_name>
    conda activate <your_env_name>
    ```
    *(See [Full Installation Guide](./docs/installation.md) for details, Pip instructions, and troubleshooting.)*

3.  **Install the Package:**
    ```bash
    pip install -e .
    ```

## Basic Inference Usage

Run inference using the `runs/augment/inference.py` script. Here's an example using the general model:

```bash
python runs/augment/inference.py \
    --model_path assets/31m2plxv-V1/model_info/best_model_370_val_f1_min=0.8028.pt \
    --config_file assets/31m2plxv-V1/model_info/model_config.json \
    --input_folder /path/to/your/audio/ \
    --output_folder /path/to/save/results/
```

*(See [Running Inference Guide](./docs/using_pretrained_model/index.md) for more details and arguments.)*

## Documentation

For complete details on installation, usage, data preparation, model performance, and troubleshooting, please refer to the **[Full Documentation Site](https://speechlabbccuny.github.io/EDANSA/)** (built from the `docs/` directory).

## Citation

If you use the code or models from this repository, please cite:

```
@inproceedings{Coban2022,
    author = "\c{C}oban, Enis Berk and Perra, Megan and Pir, Dara and Mandel, Michael I.",
    title = "EDANSA-2019: The Ecoacoustic Dataset from Arctic North Slope Alaska",
    booktitle = "Proceedings of the 7th Detection and Classification of Acoustic Scenes and Events 2022 Workshop (DCASE2022)",
    address = "Nancy, France",
    month = "November",
    year = "2022",
    abstract = "The arctic is warming at three times the rate of the global average, affecting the habitat and lifecycles of migratory species that reproduce there, like birds and caribou. Ecoacoustic monitoring can help efficiently track changes in animal phenology and behavior over large areas so that the impacts of climate change on these species can be better understood and potentially mitigated. We introduce here the Ecoacoustic Dataset from Arctic North Slope Alaska (EDANSA-2019), a dataset collected by a network of 100 autonomous recording units covering an area of 9000 square miles over the course of the 2019 summer season on the North Slope of Alaska and neighboring regions. We labeled over 27 hours of this dataset according to 28 tags with enough instances of 9 important environmental classes to train baseline convolutional recognizers. We are releasing this dataset and the corresponding baseline to the community to accelerate the recognition of these sounds and facilitate automated analyses of large-scale ecoacoustic databases."
}
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](./LICENSE.md) file for details.
