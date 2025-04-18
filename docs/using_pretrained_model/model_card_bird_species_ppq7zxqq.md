# Model Card: Bird Species Model (ppq7zxqq)

This document provides details about the pre-trained EDANSA model specifically tuned for North American arctic bird species.

## Model Details

*   **Model ID / Run Name:** `ppq7zxqq`
*   **Description:** A bioacoustics model trained to detect specific bird species commonly found in North American arctic environments.

## Assets Location

The model weights, configuration file, recommended thresholds, and raw predictions for threshold tuning are located within this repository:

*   **Model Weights:** `assets/ppq7zxqq/best_model_200_val_f1_min=0.7857_ppq7zxqq.pt`
*   **Configuration File:** `assets/ppq7zxqq/model_config_ppq7zxqq.json`
*   **Recommended Thresholds (Test Set):** `assets/ppq7zxqq/ppq7zxqq_thresholds.csv` (Contains optimal F1 thresholds derived from the test set - recommended starting points)
*   **Raw Predictions (for custom thresholding):** `assets/ppq7zxqq/all_predictions_epoch_200_ppq7zxqq.csv` (Contains raw model outputs/logits. These need transformation, e.g., via a sigmoid function, to be interpreted as probabilities between 0 and 1. Useful for advanced users wanting to determine custom thresholds).

## Performance Metrics (Test Set)

The `Short Label` corresponds to the prefix used in output CSV column headers (e.g., `pred_LALO`).

| Label (Species)           | Short Label | Test AUC | Test F1 Score | F1 Threshold | 
|---------------------------|-------------|----------|---------------|--------------|
| Lapland Longspur          | LALO        | 0.974    | 0.909         | 0.433        |
| White-crowned Sparrow     | WCSP        | 0.981    | 0.864         | 0.635        |
| American Tree Sparrow     | ATSP        | 0.991    | 0.835         | 0.222        |
| Willow Ptarmigan          | WIPT        | 0.942    | 0.802         | 0.701        |
| Savannah Sparrow          | SAVS        | 0.938    | 0.797         | 0.578        |
| Common Redpoll            | CORE        | 0.989    | 0.786         | 0.197        |

**Metric Definitions:**

*   **AUC (Area Under the ROC Curve):** Measures the model's ability to distinguish between classes across all possible thresholds. A value closer to 1.0 indicates better separability.
*   **F1 Score:** The harmonic mean of precision and recall at the specified F1 Threshold. It provides a balance between finding relevant sounds (recall) and ensuring the found sounds are indeed relevant (precision). A value closer to 1.0 is better.
*   **F1 Threshold:** The confidence score threshold optimized on the test set to achieve the reported F1 Score. Use this as a starting point for thresholding predictions.

## Intended Use

This model is intended for detecting specific bird species in passive acoustic monitoring data from relevant geographic regions (e.g., North American arctic/sub-arctic). It can be used for species occurrence studies or as input for further analysis.

## Limitations & Thresholding

*   Performance may vary depending on the specific acoustic environment, recorder type, noise levels, distance to the sound source, and regional variations in vocalizations.
*   The model's performance on species not included in the training data is unknown.
*   **Threshold Adjustment:** The optimal confidence score thresholds provided in the `ppq7zxqq_thresholds.csv` file are recommended **starting points**. Users should evaluate and potentially adjust these thresholds based on their specific project goals (e.g., prioritizing recall over precision) and test data.
*   **Advanced Thresholding:** For users wanting to perform detailed threshold analysis or optimize using different metrics, the raw model predictions (logits, which need transformation like sigmoid to become 0-1 probabilities) are available in `all_predictions_epoch_200_ppq7zxqq.csv`. 