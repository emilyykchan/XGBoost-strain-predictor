from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "rotvelx",
    "rotvely",
    "rotvelz",
    "rotaccx",
    "rotaccy",
    "rotaccz",
]

FEATURE_LABELS = [
    "Rotational velocity resultant range",
    "Square root of maximum rotational velocity resultant",
    "Rotational acceleration resultant range",
    "Square root of maximum rotational acceleration resultant",
]


def read_impact_csv(path: str | Path) -> pd.DataFrame:
    """
    Read and validate one head-impact CSV.

    One CSV represents one impact.
    Each row represents one time point.
    """

    path = Path(path)

    try:
        dataframe = pd.read_csv(path)

    except pd.errors.EmptyDataError as error:
        raise ValueError("The CSV is empty.") from error

    except pd.errors.ParserError as error:
        raise ValueError("The CSV could not be parsed.") from error

    if dataframe.empty:
        raise ValueError("The CSV contains no time points.")

    # Ignore differences in capitalisation and surrounding spaces.
    dataframe.columns = [
        str(column).strip().lower()
        for column in dataframe.columns
    ]

    if len(set(dataframe.columns)) != len(dataframe.columns):
        raise ValueError(
            "Duplicate column names were found after normalisation."
        )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        missing_text = ", ".join(missing_columns)

        raise ValueError(
            f"Missing required column(s): {missing_text}"
        )

    if len(dataframe) < 2:
        raise ValueError(
            "An impact must contain at least two time points."
        )

    # Convert only the required columns to numeric values.
    for column in REQUIRED_COLUMNS:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

        values = dataframe[column].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(
                f"Column '{column}' contains missing or "
                "non-numeric values."
            )

    return dataframe


def extract_impact_features(
    path: str | Path,
) -> dict[str, object]:
    """
    Convert one impact time series into one four-feature vector.

    Feature order:

    1. Range of rotational-velocity resultant
    2. Square root of maximum rotational-velocity resultant
    3. Range of rotational-acceleration resultant
    4. Square root of maximum rotational-acceleration resultant
    """

    dataframe = read_impact_csv(path)

    rotational_velocity_resultant = np.sqrt(
        dataframe["rotvelx"].to_numpy(dtype=float) ** 2
        + dataframe["rotvely"].to_numpy(dtype=float) ** 2
        + dataframe["rotvelz"].to_numpy(dtype=float) ** 2
    )

    rotational_acceleration_resultant = np.sqrt(
        dataframe["rotaccx"].to_numpy(dtype=float) ** 2
        + dataframe["rotaccy"].to_numpy(dtype=float) ** 2
        + dataframe["rotaccz"].to_numpy(dtype=float) ** 2
    )

    feature_values = np.array(
        [
            np.ptp(rotational_velocity_resultant),
            np.sqrt(
                abs(np.max(rotational_velocity_resultant))
            ),
            np.ptp(rotational_acceleration_resultant),
            np.sqrt(
                abs(np.max(rotational_acceleration_resultant))
            ),
        ],
        dtype=float,
    )

    interpretation = (
        "rotvelx/y/z interpreted as rotational velocity; "
        "rotaccx/y/z interpreted as rotational acceleration. "
        "Resultants calculated from the three axes."
    )

    return {
        "file_name": Path(path).name,
        "time_points": len(dataframe),
        "detected_columns": ", ".join(dataframe.columns),
        "interpretation": interpretation,
        "feature_labels": FEATURE_LABELS,
        "feature_values": feature_values,
    }