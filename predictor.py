from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import xgboost as xgb
from xgboost.core import XGBoostError


EXPECTED_FEATURE_COUNT = 4

DEFAULT_MANIFEST_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "manifest.json"
)


class ModelConfigurationError(Exception):
    """Raised when the model manifest is missing or invalid."""


@dataclass(frozen=True)
class ModelSpec:
    """Description of one regional prediction model."""

    model_id: str
    file_name: str
    display_name: str
    short_name: str
    primary: bool


class RegionalPredictor:
    """
    Load and run all available regional XGBoost models.

    Model filenames, display names and display order are defined
    inside models/manifest.json.

    Missing regional model files are reported but do not prevent
    available models from running.
    """

    def __init__(
        self,
        manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.models_directory = self.manifest_path.parent

        self.model_version = ""
        self.prediction_target = ""
        self.feature_order: list[str] = []

        self.specs: list[ModelSpec] = []

        # Native XGBoost Booster models.
        self.models: dict[str, xgb.Booster] = {}

        # Model ID → loading error.
        self.load_errors: dict[str, str] = {}

        self._load_manifest()
        self._load_available_models()

    def _load_manifest(self) -> None:
        """Read and validate models/manifest.json."""

        if not self.manifest_path.exists():
            raise ModelConfigurationError(
                "Model manifest not found at: "
                f"{self.manifest_path}"
            )

        try:
            with self.manifest_path.open(
                "r",
                encoding="utf-8",
            ) as manifest_file:
                manifest = json.load(manifest_file)

        except json.JSONDecodeError as error:
            raise ModelConfigurationError(
                "The model manifest is not valid JSON."
            ) from error

        except OSError as error:
            raise ModelConfigurationError(
                "The model manifest could not be read."
            ) from error

        if not isinstance(manifest, dict):
            raise ModelConfigurationError(
                "The model manifest must contain a JSON object."
            )

        self.model_version = str(
            manifest.get("model_version", "")
        ).strip()

        self.prediction_target = str(
            manifest.get("prediction_target", "")
        ).strip()

        feature_order = manifest.get("feature_order")

        if not isinstance(feature_order, list):
            raise ModelConfigurationError(
                "The manifest must contain a feature_order list."
            )

        if len(feature_order) != EXPECTED_FEATURE_COUNT:
            raise ModelConfigurationError(
                "The manifest feature_order must contain exactly "
                f"{EXPECTED_FEATURE_COUNT} features."
            )

        self.feature_order = [
            str(feature).strip()
            for feature in feature_order
        ]

        model_entries = manifest.get("models")

        if not isinstance(model_entries, list):
            raise ModelConfigurationError(
                "The manifest must contain a models list."
            )

        if not model_entries:
            raise ModelConfigurationError(
                "The model manifest contains no model entries."
            )

        model_ids: set[str] = set()
        file_names: set[str] = set()
        primary_count = 0

        for entry in model_entries:
            if not isinstance(entry, dict):
                raise ModelConfigurationError(
                    "Every model entry must be a JSON object."
                )

            required_fields = [
                "id",
                "file",
                "display_name",
                "short_name",
            ]

            missing_fields = [
                field
                for field in required_fields
                if field not in entry
            ]

            if missing_fields:
                raise ModelConfigurationError(
                    "A model entry is missing required field(s): "
                    + ", ".join(missing_fields)
                )

            model_id = str(entry["id"]).strip()
            file_name = str(entry["file"]).strip()
            display_name = str(
                entry["display_name"]
            ).strip()
            short_name = str(
                entry["short_name"]
            ).strip()
            primary = bool(
                entry.get("primary", False)
            )

            if not model_id:
                raise ModelConfigurationError(
                    "A model entry has an empty id."
                )

            if not file_name:
                raise ModelConfigurationError(
                    f"Model '{model_id}' has an empty filename."
                )

            if not display_name:
                raise ModelConfigurationError(
                    f"Model '{model_id}' has an empty display name."
                )

            if model_id in model_ids:
                raise ModelConfigurationError(
                    f"Duplicate model id: {model_id}"
                )

            if file_name in file_names:
                raise ModelConfigurationError(
                    f"Duplicate model filename: {file_name}"
                )

            if primary:
                primary_count += 1

            model_ids.add(model_id)
            file_names.add(file_name)

            self.specs.append(
                ModelSpec(
                    model_id=model_id,
                    file_name=file_name,
                    display_name=display_name,
                    short_name=short_name,
                    primary=primary,
                )
            )

        if primary_count != 1:
            raise ModelConfigurationError(
                "Exactly one model must have primary set to true."
            )

    def _load_available_models(self) -> None:
        """
        Load every available JSON model using XGBoost Booster.

        Models missing from the folder are recorded as unavailable,
        rather than stopping the entire application.
        """

        for spec in self.specs:
            model_path = (
                self.models_directory
                / spec.file_name
            )

            if not model_path.exists():
                self.load_errors[spec.model_id] = (
                    f"File not found: {spec.file_name}"
                )
                continue

            booster = xgb.Booster()

            try:
                booster.load_model(
                    str(model_path)
                )

                feature_count = int(
                    booster.num_features()
                )

                if feature_count != EXPECTED_FEATURE_COUNT:
                    raise ValueError(
                        f"Model expects {feature_count} features; "
                        f"{EXPECTED_FEATURE_COUNT} are required."
                    )

            except (
                XGBoostError,
                ValueError,
                TypeError,
                AttributeError,
                OSError,
            ) as error:
                self.load_errors[spec.model_id] = str(error)
                continue

            self.models[spec.model_id] = booster

    @property
    def loaded_specs(self) -> list[ModelSpec]:
        """Return successfully loaded models in manifest order."""

        return [
            spec
            for spec in self.specs
            if spec.model_id in self.models
        ]

    @property
    def primary_spec(self) -> ModelSpec:
        """Return the model marked as the primary output."""

        for spec in self.specs:
            if spec.primary:
                return spec

        raise ModelConfigurationError(
            "No primary model is defined."
        )

    @property
    def primary_model_loaded(self) -> bool:
        """Return whether the primary model is loaded."""

        return (
            self.primary_spec.model_id
            in self.models
        )

    @property
    def loaded_model_count(self) -> int:
        """Return the number of loaded model files."""

        return len(self.models)

    @property
    def total_model_count(self) -> int:
        """Return the number of models listed in the manifest."""

        return len(self.specs)

    def get_spec(
        self,
        model_id: str,
    ) -> ModelSpec:
        """Find a model specification by model ID."""

        for spec in self.specs:
            if spec.model_id == model_id:
                return spec

        raise KeyError(
            f"Unknown model id: {model_id}"
        )

    def _prepare_features(
        self,
        feature_values: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Validate one impact's four-feature vector."""

        features = np.asarray(
            feature_values,
            dtype=float,
        ).reshape(-1)

        if features.size != EXPECTED_FEATURE_COUNT:
            raise ValueError(
                "Prediction requires exactly "
                f"{EXPECTED_FEATURE_COUNT} features."
            )

        if not np.isfinite(features).all():
            raise ValueError(
                "Prediction features contain missing "
                "or non-finite values."
            )

        return features.reshape(
            1,
            EXPECTED_FEATURE_COUNT,
        )

    @staticmethod
    def _prediction_iteration_range(
        booster: xgb.Booster,
    ) -> tuple[int, int]:
        """
        Use the saved best iteration where available.

        XGBoost stores best_iteration as a zero-based tree index.
        The upper end of iteration_range is exclusive.
        """

        best_iteration = booster.attr(
            "best_iteration"
        )

        if best_iteration is None:
            return 0, 0

        try:
            final_iteration = int(
                best_iteration
            ) + 1
        except ValueError:
            return 0, 0

        return 0, final_iteration

    def predict_all(
        self,
        feature_values: Sequence[float] | np.ndarray,
    ) -> tuple[dict[str, float], dict[str, str]]:
        """
        Predict all available regions for one impact.

        Returns:
            predictions:
                Model ID → predicted strain value.

            errors:
                Model ID → prediction error message.
        """

        feature_matrix = self._prepare_features(
            feature_values
        )

        impact_matrix = xgb.DMatrix(
            feature_matrix
        )

        predictions: dict[str, float] = {}
        errors: dict[str, str] = {}

        for spec in self.loaded_specs:
            booster = self.models[
                spec.model_id
            ]

            try:
                iteration_range = (
                    self._prediction_iteration_range(
                        booster
                    )
                )

                output = booster.predict(
                    impact_matrix,
                    iteration_range=iteration_range,
                )

                output_array = np.asarray(
                    output,
                    dtype=float,
                ).reshape(-1)

                if output_array.size != 1:
                    raise RuntimeError(
                        "The model returned an unexpected "
                        "prediction shape."
                    )

                predicted_value = float(
                    output_array[0]
                )

                if not np.isfinite(predicted_value):
                    raise RuntimeError(
                        "The model returned a non-finite value."
                    )

                predictions[
                    spec.model_id
                ] = predicted_value

            except (
                XGBoostError,
                ValueError,
                RuntimeError,
            ) as error:
                errors[spec.model_id] = str(error)

        return predictions, errors