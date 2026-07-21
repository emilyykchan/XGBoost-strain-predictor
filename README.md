# XGBoost Brain Strain Predictor

A desktop app for rapid prediction of whole-brain and regional brain strain from head-impact kinematics using pretrained XGBoost models.

> **Research use only.** Outputs are model predictions, not clinical diagnoses, concussion probabilities, or direct measurements of injury.

## Download

Download the latest macOS version from the [Releases page](../../releases/latest).

After downloading:

1. Unzip the release.
2. Move **XGBoost Regional Brain Strain Predictor.app** to `Applications`.
3. Open the app.
4. Load one or more impact CSV files.
5. Check the **Input validation** tab.
6. Select **Run regional predictions**.
7. View Whole Brain as the primary result and the other available regions in the results table.

## Input CSV format

Each CSV represents **one head impact**. Each row represents one time point in that impact's kinematic time series.

The following columns are required:

| Column | Quantity | Unit |
|---|---|---|
| `rotvelx` | Rotational velocity, x-axis | rad/s |
| `rotvely` | Rotational velocity, y-axis | rad/s |
| `rotvelz` | Rotational velocity, z-axis | rad/s |
| `rotaccx` | Rotational acceleration, x-axis | rad/s² |
| `rotaccy` | Rotational acceleration, y-axis | rad/s² |
| `rotaccz` | Rotational acceleration, z-axis | rad/s² |

Additional columns are allowed and ignored. The app does not currently convert units.
A complete example is provided in [`sample_data/sample_impact_1.csv`](sample_data/sample_impact_1.csv).

## What the app calculates

For each impact, the app calculates rotational-velocity and rotational-acceleration resultants from the three axes, then extracts four features in the order required by the trained models:

1. Range of rotational-velocity resultant.
2. Square root of maximum rotational-velocity resultant.
3. Range of rotational-acceleration resultant.
4. Square root of maximum rotational-acceleration resultant.

The app displays **Whole Brain** as the primary result, followed by predictions for the available anatomical regions. Whole Brain is a broad summary and is not an average of the regional predictions.

## Running from source

This section is for backup only. Users should download the packaged app from **Releases**.

Python 3.12 is recommended.

```bash
git clone https://github.com/emilyykchan/XGBoost-strain-predictor.git
cd XGBoost-strain-predictor

python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install pyside6 pandas numpy xgboost

python app.py
```

The app uses XGBoost's native `Booster` interface and does not require scikit-learn.

## Out-of-scopes

- The app does not resample, filter, rotate, or otherwise preprocess signals.
- Reliability depends on the data, preprocessing, finite-element targets, and population used to train the models.
- Predictions outside the training distribution may be unreliable.

## Citation

Please cite the following study:

> Chan, E. Y. K., Yu, X., Qin, C., & Ghajari, M. (2025). Balancing efficiency and accuracy: Extreme gradient boosting and neural networks for near real-time brain deformation prediction in sports collisions. Engineering Applications of Artificial Intelligence, 162, 112489. https://doi.org/10.1016/j.engappai.2025.112489

## Licence

This project is distributed under the terms provided in [`LICENSE`](LICENSE).
