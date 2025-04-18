# Model Card: EDANSA Bioacoustics Model (31m2plxv-V1)

This document provides details about the primary pre-trained EDANSA model.

## Model Details

*   **Model ID:** `31m2plxv-V1`
*   **Description:** A general-purpose bioacoustics model trained to detect various sound event categories relevant to ecological monitoring.

## Assets Location

The model weights, configuration file, recommended thresholds, and raw predictions for threshold tuning are located within this repository:

*   **Model Weights:** `assets/31m2plxv-V1/model_info/best_model_370_val_f1_min=0.8028.pt`
*   **Configuration File:** `assets/31m2plxv-V1/model_info/model_config.json`
*   **Recommended Thresholds (Validation Set):** `assets/31m2plxv-V1/model_info/thresholds_31m2plxv_epoch370.csv` (Contains optimal thresholds derived from the validation set - recommended starting points for users)
*   **Raw Predictions (for custom thresholding):** `assets/31m2plxv-V1/model_info/datasetV5.4.10_modelcard.csv` (Contains raw model outputs, often called logits. These need transformation, e.g., via a sigmoid function, to be interpreted as probabilities between 0 and 1. Useful for advanced users wanting to determine custom thresholds).

## Performance Metrics (Test Set)

The following metrics were evaluated on an independent test set, providing an estimate of the model's performance on unseen data. The F1 scores reported here were calculated by applying the optimal thresholds. The `Short Label` corresponds to the prefix used in output CSV column headers (e.g., `pred_Bio`).

| Label (Class)                 | Short Label | Test AUC | Test F1 Score |
|-------------------------------|-------------|----------|---------------|
| Biophony                      | Bio         | 0.973    | 0.935         |
| Bird (Generic)                | Bird        | 0.973    | 0.920         |
| Songbird                      | SongB       | 0.957    | 0.829         |
| Duck/Goose/Swan               | DGS         | 0.925    | 0.591         |
| Grouse                        | Grous       | 0.948    | 0.763         |
| Insect                        | Bug         | 0.969    | 0.902         |
| Anthropophony                 | Anth        | 0.995    | 0.976         |
| Aircraft                      | Airc        | 0.967    | 0.850         |
| Silence                       | Sil         | 0.957    | 0.789         |

**Metric Definitions:**

*   **AUC (Area Under the ROC Curve):** Measures the model's ability to distinguish between classes across all possible thresholds. A value closer to 1.0 indicates better separability.
*   **F1 Score:** The harmonic mean of precision and recall, calculated using the thresholds from the validation set CSV. It provides a balance between finding relevant sounds (recall) and ensuring the found sounds are indeed relevant (precision). A value closer to 1.0 is better.

## Intended Use

This model is intended for detecting bioacoustic events in passive acoustic monitoring data. It can serve as a baseline classifier or feature extractor for ecological studies.

## Limitations & Thresholding

*   Performance may vary depending on the specific acoustic environment, recorder type, noise levels, and distance to the sound source.
*   The model's performance on classes not included in the training data is unknown.
*   **Threshold Adjustment:** The optimal confidence score thresholds provided in the `thresholds_31m2plxv_epoch370.csv` file (derived from the validation set) are recommended **starting points**. Users should evaluate and potentially adjust these thresholds based on their specific project goals (e.g., prioritizing recall over precision) and validation data.
*   **Advanced Thresholding:** For users wanting to perform detailed threshold analysis or optimize using different metrics, the raw model predictions (logits, which need transformation like sigmoid to become 0-1 probabilities) are available in `datasetV5.4.10_modelcard.csv`. 