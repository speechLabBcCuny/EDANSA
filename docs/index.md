# Welcome to the EDANSA Bioacoustics Model Documentation

This documentation provides information on using the pre-trained bioacoustic models developed as part of the EDANSA project.

## Overview

Passive Acoustic Monitoring (PAM) generates vast amounts of audio data, requiring efficient automated methods for analysis. The models presented here are designed to detect various bioacoustic events (like general bird song, insect sounds, or anthropophony) and specific bird species calls within audio recordings.

We provide two primary pre-trained models ready for use:

1.  **General Bioacoustics Model (`31m2plxv-V1`):** Detects broader categories of sound events relevant to ecological monitoring.
2.  **Bird Species Model (`ppq7zxqq`):** Focused on identifying specific bird species commonly found in North American arctic environments.

## Getting Started

*   **Installation:** Begin by following the [Installation Guide](./installation.md) to set up the necessary environment and dependencies.
*   **Using a Model:** Learn how to run inference with the pre-trained models:
    *   [Running Inference](./using_pretrained_model/index.md): Command-line arguments and examples.
    *   [Providing Audio Data](./using_pretrained_model/providing_audio_data.md): How to format your input.
    *   [Understanding Results](./using_pretrained_model/output_explanation.md): Explanation of the output CSV files.
*   **Model Details:** Find specific performance metrics and details for each model:
    *   [General Model Card](./using_pretrained_model/model_card_general_31m2plxv-V1.md)
    *   [Bird Species Model Card](./using_pretrained_model/model_card_bird_species_ppq7zxqq.md)

## Troubleshooting

Encountering issues? Check the [Troubleshooting Guide](./common/troubleshooting.md). 