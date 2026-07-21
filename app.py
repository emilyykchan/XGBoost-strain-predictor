from __future__ import annotations

import csv
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from kinematics import extract_impact_features
from predictor import (
    ModelConfigurationError,
    RegionalPredictor,
)


class MainWindow(QMainWindow):
    """Main window for the XGBoost strain predictor."""

    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle(
            "XGBoost Regional Brain Strain Predictor"
        )
        self.resize(1500, 760)

        self.loaded_files: list[Path] = []

        # Validated information for each impact.
        self.impact_metadata: dict[
            Path,
            dict[str, object],
        ] = {}

        self.impact_features: dict[
            Path,
            object,
        ] = {}

        # Prediction results for exporting.
        self.impact_predictions: dict[
            Path,
            dict[str, float],
        ] = {}

        self.impact_prediction_errors: dict[
            Path,
            dict[str, str],
        ] = {}

        self.impact_statuses: dict[
            Path,
            str,
        ] = {}

        self.prediction_attempted_paths: set[
            Path
        ] = set()

        self.predictor: RegionalPredictor | None = None
        self.predictor_error: str | None = None

        try:
            self.predictor = RegionalPredictor()

        except ModelConfigurationError as error:
            self.predictor_error = str(error)

        # Maps model IDs to prediction-table columns.
        self.region_column_by_id: dict[
            str,
            int,
        ] = {}

        title_label = QLabel(
            "XGBoost Regional Brain Strain Predictor"
        )
        title_label.setStyleSheet(
            "font-size: 26px; font-weight: bold;"
        )

        description_label = QLabel(
            "Each CSV represents one head impact. "
            "Each row represents one time point in the "
            "impact time history."
        )
        description_label.setWordWrap(True)

        required_columns_label = QLabel(
            "Required columns: "
            "rotvelx, rotvely, rotvelz, "
            "rotaccx, rotaccy, rotaccz"
        )
        required_columns_label.setStyleSheet(
            "font-weight: bold;"
        )
        required_columns_label.setWordWrap(True)

        units_label = QLabel(
            "Required units: rotational velocity in rad/s and "
            "rotational acceleration in rad/s²."
        )
        units_label.setWordWrap(True)

        self.model_status_label = QLabel()
        self.update_model_status_label()

        self.primary_result_label = QLabel(
            "Primary result: Whole Brain prediction "
            "has not been run."
        )
        self.primary_result_label.setStyleSheet(
            "font-size: 18px; font-weight: bold; "
            "padding: 10px;"
        )
        self.primary_result_label.setWordWrap(True)

        self.status_label = QLabel(
            "No impact files loaded."
        )

        self.load_button = QPushButton(
            "Load impact CSV files"
        )
        self.load_button.setMinimumHeight(40)
        self.load_button.clicked.connect(
            self.load_csv_files
        )

        self.run_button = QPushButton(
            "Run regional predictions"
        )
        self.run_button.setMinimumHeight(40)
        self.run_button.clicked.connect(
            self.run_predictions
        )

        self.export_button = QPushButton(
            "Export predictions CSV"
        )
        self.export_button.setMinimumHeight(40)
        self.export_button.clicked.connect(
            self.export_predictions_csv
        )

        self.clear_button = QPushButton(
            "Clear files"
        )
        self.clear_button.setMinimumHeight(40)
        self.clear_button.clicked.connect(
            self.clear_files
        )

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.run_button)
        button_layout.addWidget(self.export_button)
        button_layout.addWidget(self.clear_button)
        button_layout.addStretch()

        self.validation_table = QTableWidget()
        self.configure_validation_table()

        self.prediction_table = QTableWidget()
        self.configure_prediction_table()

        self.prediction_table.itemSelectionChanged.connect(
            self.update_primary_result_label
        )

        tabs = QTabWidget()
        tabs.addTab(
            self.validation_table,
            "Input validation",
        )
        tabs.addTab(
            self.prediction_table,
            "Regional predictions",
        )

        main_layout = QVBoxLayout()
        main_layout.addWidget(title_label)
        main_layout.addWidget(description_label)
        main_layout.addWidget(required_columns_label)
        main_layout.addWidget(units_label)
        main_layout.addWidget(self.model_status_label)
        main_layout.addSpacing(10)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.status_label)
        main_layout.addWidget(self.primary_result_label)
        main_layout.addWidget(tabs)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)

        self.setCentralWidget(central_widget)

        self.update_button_state()

    def configure_validation_table(self) -> None:
        """Configure the CSV interpretation table."""

        headers = [
            "Impact file",
            "Time points",
            "Detected columns",
            "Interpretation",
            "Velocity range",
            "√ maximum velocity",
            "Acceleration range",
            "√ maximum acceleration",
            "Status",
        ]

        self.validation_table.setColumnCount(
            len(headers)
        )

        self.validation_table.setHorizontalHeaderLabels(
            headers
        )

        self.configure_common_table_options(
            self.validation_table
        )

        header = (
            self.validation_table.horizontalHeader()
        )

        header.setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )

        header.setSectionResizeMode(
            2,
            QHeaderView.ResizeMode.Interactive,
        )

        header.setSectionResizeMode(
            3,
            QHeaderView.ResizeMode.Stretch,
        )

        header.setSectionResizeMode(
            8,
            QHeaderView.ResizeMode.Stretch,
        )

        self.validation_table.setColumnWidth(
            2,
            300,
        )

        self.validation_table.setColumnWidth(
            3,
            420,
        )

    def configure_prediction_table(self) -> None:
        """Configure the regional prediction table."""

        loaded_specs = []

        if self.predictor is not None:
            loaded_specs = self.predictor.loaded_specs

        headers = ["Impact file"]

        for column_number, spec in enumerate(
            loaded_specs,
            start=1,
        ):
            headers.append(spec.display_name)

            self.region_column_by_id[
                spec.model_id
            ] = column_number

        headers.append("Status")

        self.prediction_status_column = (
            len(headers) - 1
        )

        self.prediction_table.setColumnCount(
            len(headers)
        )

        self.prediction_table.setHorizontalHeaderLabels(
            headers
        )

        self.configure_common_table_options(
            self.prediction_table
        )

        header = (
            self.prediction_table.horizontalHeader()
        )

        header.setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )

        header.setSectionResizeMode(
            self.prediction_status_column,
            QHeaderView.ResizeMode.Stretch,
        )

        # Emphasise the Whole Brain column.
        if self.predictor is not None:
            primary_spec = self.predictor.primary_spec

            primary_column = (
                self.region_column_by_id.get(
                    primary_spec.model_id
                )
            )

            if primary_column is not None:
                header_item = (
                    self.prediction_table
                    .horizontalHeaderItem(
                        primary_column
                    )
                )

                if header_item is not None:
                    font = QFont(
                        header_item.font()
                    )
                    font.setBold(True)
                    header_item.setFont(font)

    def configure_common_table_options(
        self,
        table: QTableWidget,
    ) -> None:
        """Apply shared behaviour to both tables."""

        table.setAlternatingRowColors(True)
        table.setSortingEnabled(True)
        table.setWordWrap(False)

        table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )

        table.setSelectionMode(
            QTableWidget.SelectionMode.SingleSelection
        )

    def update_model_status_label(self) -> None:
        """Display which model files were loaded."""

        if self.predictor is None:
            self.model_status_label.setText(
                "Models unavailable: "
                f"{self.predictor_error}"
            )
            self.model_status_label.setStyleSheet(
                "font-weight: bold;"
            )
            return

        loaded_count = (
            self.predictor.loaded_model_count
        )

        total_count = (
            self.predictor.total_model_count
        )

        if self.predictor.primary_model_loaded:
            primary_text = "Whole Brain loaded"
        else:
            primary_text = "Whole Brain unavailable"

        self.model_status_label.setText(
            f"Models loaded: {loaded_count}/{total_count}. "
            f"Primary model: {primary_text}."
        )

        self.model_status_label.setStyleSheet(
            "font-weight: bold;"
        )

        if self.predictor.load_errors:
            error_lines = []

            for model_id, error_message in (
                self.predictor.load_errors.items()
            ):
                spec = self.predictor.get_spec(
                    model_id
                )

                error_lines.append(
                    f"{spec.display_name}: "
                    f"{error_message}"
                )

            self.model_status_label.setToolTip(
                "\n".join(error_lines)
            )

    def load_csv_files(self) -> None:
        """Select and validate multiple impact CSV files."""

        selected_files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select head-impact CSV files",
            "",
            "CSV files (*.csv)",
        )

        if not selected_files:
            return

        self.validation_table.setSortingEnabled(
            False
        )

        self.prediction_table.setSortingEnabled(
            False
        )

        added_count = 0

        for selected_file in selected_files:
            file_path = Path(
                selected_file
            ).resolve()

            if file_path in self.loaded_files:
                continue

            self.loaded_files.append(file_path)

            self.add_impact_to_tables(
                file_path
            )

            added_count += 1

        self.validation_table.setSortingEnabled(
            True
        )

        self.prediction_table.setSortingEnabled(
            True
        )

        if added_count == 0:
            self.status_label.setText(
                "No new files were added. "
                "The selected files were already loaded."
            )
        else:
            self.update_status_label()

        self.update_button_state()

    def add_impact_to_tables(
        self,
        file_path: Path,
    ) -> None:
        """Add one impact to both application tables."""

        validation_row = (
            self.validation_table.rowCount()
        )

        self.validation_table.insertRow(
            validation_row
        )

        prediction_row = (
            self.prediction_table.rowCount()
        )

        self.prediction_table.insertRow(
            prediction_row
        )

        validation_values = [
            file_path.name,
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
        ]

        prediction_values = [
            ""
            for _ in range(
                self.prediction_table.columnCount()
            )
        ]

        prediction_values[0] = file_path.name

        try:
            result = extract_impact_features(
                file_path
            )

            feature_values = result[
                "feature_values"
            ]

            self.impact_metadata[
                file_path
            ] = result

            self.impact_features[
                file_path
            ] = feature_values

            status_text = "Ready for prediction"

            self.impact_statuses[
                file_path
            ] = status_text

            validation_values = [
                result["file_name"],
                str(result["time_points"]),
                result["detected_columns"],
                result["interpretation"],
                f"{feature_values[0]:.6f}",
                f"{feature_values[1]:.6f}",
                f"{feature_values[2]:.6f}",
                f"{feature_values[3]:.6f}",
                status_text,
            ]

            prediction_values[
                self.prediction_status_column
            ] = status_text

        except (
            ValueError,
            OSError,
            UnicodeDecodeError,
        ) as error:
            error_text = f"Invalid: {error}"

            self.impact_statuses[
                file_path
            ] = error_text

            validation_values[-1] = error_text

            prediction_values[
                self.prediction_status_column
            ] = error_text

        self.populate_table_row(
            self.validation_table,
            validation_row,
            validation_values,
            file_path,
        )

        self.populate_table_row(
            self.prediction_table,
            prediction_row,
            prediction_values,
            file_path,
        )

    def populate_table_row(
        self,
        table: QTableWidget,
        row_number: int,
        values: list[object],
        file_path: Path,
    ) -> None:
        """Write one complete read-only table row."""

        for column_number, value in enumerate(
            values
        ):
            item = self.create_read_only_item(
                value
            )

            if column_number == 0:
                item.setData(
                    Qt.ItemDataRole.UserRole,
                    str(file_path),
                )

            table.setItem(
                row_number,
                column_number,
                item,
            )

    @staticmethod
    def create_read_only_item(
        value: object,
        tooltip: str | None = None,
    ) -> QTableWidgetItem:
        """Create a non-editable table item."""

        item = QTableWidgetItem(
            str(value)
        )

        item.setFlags(
            item.flags()
            & ~Qt.ItemFlag.ItemIsEditable
        )

        item.setToolTip(
            tooltip
            if tooltip is not None
            else str(value)
        )

        return item

    def run_predictions(self) -> None:
        """Predict every available region for every valid impact."""

        if self.predictor is None:
            QMessageBox.critical(
                self,
                "Models unavailable",
                self.predictor_error
                or "The models are unavailable.",
            )
            return

        if not self.predictor.models:
            QMessageBox.critical(
                self,
                "No models loaded",
                "No regional model files could be loaded.",
            )
            return

        if not self.impact_features:
            QMessageBox.information(
                self,
                "No valid impacts",
                "Load at least one valid impact CSV first.",
            )
            return

        self.prediction_table.setSortingEnabled(
            False
        )

        successful_impacts = 0
        partial_impacts = 0
        failed_impacts = 0

        for row_number in range(
            self.prediction_table.rowCount()
        ):
            file_item = self.prediction_table.item(
                row_number,
                0,
            )

            if file_item is None:
                continue

            stored_path = file_item.data(
                Qt.ItemDataRole.UserRole
            )

            if not stored_path:
                continue

            file_path = Path(
                str(stored_path)
            )

            feature_values = (
                self.impact_features.get(
                    file_path
                )
            )

            # Invalid impacts have no feature vector.
            if feature_values is None:
                continue

            predictions, errors = (
                self.predictor.predict_all(
                    feature_values
                )
            )

            self.impact_predictions[
                file_path
            ] = predictions

            self.impact_prediction_errors[
                file_path
            ] = errors

            self.prediction_attempted_paths.add(
                file_path
            )

            for model_id, predicted_value in (
                predictions.items()
            ):
                column_number = (
                    self.region_column_by_id.get(
                        model_id
                    )
                )

                if column_number is None:
                    continue

                prediction_item = (
                    self.create_read_only_item(
                        f"{predicted_value:.6f}",
                        f"{predicted_value:.10f}",
                    )
                )

                self.prediction_table.setItem(
                    row_number,
                    column_number,
                    prediction_item,
                )

            for model_id, error_message in (
                errors.items()
            ):
                column_number = (
                    self.region_column_by_id.get(
                        model_id
                    )
                )

                if column_number is None:
                    continue

                error_item = (
                    self.create_read_only_item(
                        "Error",
                        error_message,
                    )
                )

                self.prediction_table.setItem(
                    row_number,
                    column_number,
                    error_item,
                )

            if predictions and not errors:
                status_text = "Prediction complete"
                successful_impacts += 1

            elif predictions and errors:
                status_text = (
                    "Partial prediction: "
                    f"{len(errors)} model(s) failed"
                )
                partial_impacts += 1

            else:
                status_text = "Prediction failed"
                failed_impacts += 1

            self.impact_statuses[
                file_path
            ] = status_text

            status_item = (
                self.create_read_only_item(
                    status_text
                )
            )

            self.prediction_table.setItem(
                row_number,
                self.prediction_status_column,
                status_item,
            )

        self.prediction_table.setSortingEnabled(
            True
        )

        self.status_label.setText(
            "Prediction finished: "
            f"{successful_impacts} successful, "
            f"{partial_impacts} partial, "
            f"{failed_impacts} failed."
        )

        if (
            self.prediction_table.rowCount() > 0
            and not self.prediction_table
            .selectionModel()
            .hasSelection()
        ):
            self.prediction_table.selectRow(0)

        self.update_primary_result_label()
        self.update_button_state()

    def export_predictions_csv(self) -> None:
        """Export features and regional predictions to one CSV."""

        if self.predictor is None:
            QMessageBox.critical(
                self,
                "Models unavailable",
                "The model configuration is unavailable.",
            )
            return

        if not self.prediction_attempted_paths:
            QMessageBox.information(
                self,
                "No predictions",
                "Run regional predictions before exporting.",
            )
            return

        selected_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export regional predictions",
            "regional_strain_predictions.csv",
            "CSV files (*.csv)",
        )

        if not selected_path:
            return

        output_path = Path(selected_path)

        if output_path.suffix.lower() != ".csv":
            output_path = output_path.with_suffix(
                ".csv"
            )

        loaded_specs = self.predictor.loaded_specs

        feature_headers = [
            f"Feature: {feature_name}"
            for feature_name in self.predictor.feature_order
        ]

        prediction_headers = [
            f"Prediction: {spec.display_name}"
            for spec in loaded_specs
        ]

        headers = [
            "Impact file",
            "Time points",
            "Model version",
            *feature_headers,
            *prediction_headers,
            "Status",
        ]

        try:
            with output_path.open(
                "w",
                newline="",
                encoding="utf-8-sig",
            ) as output_file:
                writer = csv.writer(output_file)

                writer.writerow(headers)

                for file_path in self.loaded_files:
                    metadata = self.impact_metadata.get(
                        file_path
                    )

                    if metadata is None:
                        time_points = ""
                        feature_cells = [
                            "",
                            "",
                            "",
                            "",
                        ]
                    else:
                        time_points = metadata.get(
                            "time_points",
                            "",
                        )

                        feature_values = metadata.get(
                            "feature_values",
                            [],
                        )

                        feature_cells = [
                            f"{float(value):.10f}"
                            for value in feature_values
                        ]

                        while len(feature_cells) < 4:
                            feature_cells.append("")

                        feature_cells = feature_cells[:4]

                    predictions = (
                        self.impact_predictions.get(
                            file_path,
                            {},
                        )
                    )

                    prediction_cells = [
                        (
                            f"{predictions[spec.model_id]:.10f}"
                            if spec.model_id in predictions
                            else ""
                        )
                        for spec in loaded_specs
                    ]

                    status_text = (
                        self.impact_statuses.get(
                            file_path,
                            "",
                        )
                    )

                    row = [
                        file_path.name,
                        time_points,
                        self.predictor.model_version,
                        *feature_cells,
                        *prediction_cells,
                        status_text,
                    ]

                    writer.writerow(row)

        except OSError as error:
            QMessageBox.critical(
                self,
                "Export failed",
                f"The CSV could not be saved:\n{error}",
            )
            return

        QMessageBox.information(
            self,
            "Export complete",
            f"Predictions were saved to:\n{output_path}",
        )

    def update_primary_result_label(self) -> None:
        """Show Whole Brain as the selected impact's headline."""

        if self.predictor is None:
            self.primary_result_label.setText(
                "Primary result unavailable."
            )
            return

        primary_spec = self.predictor.primary_spec

        primary_column = (
            self.region_column_by_id.get(
                primary_spec.model_id
            )
        )

        if primary_column is None:
            self.primary_result_label.setText(
                "Primary result unavailable: "
                "Whole Brain model is not loaded."
            )
            return

        row_number = (
            self.prediction_table.currentRow()
        )

        if row_number < 0:
            self.primary_result_label.setText(
                "Primary result: select an impact "
                "from the regional predictions table."
            )
            return

        file_item = self.prediction_table.item(
            row_number,
            0,
        )

        prediction_item = (
            self.prediction_table.item(
                row_number,
                primary_column,
            )
        )

        file_name = (
            file_item.text()
            if file_item is not None
            else "Selected impact"
        )

        prediction_text = (
            prediction_item.text().strip()
            if prediction_item is not None
            else ""
        )

        if prediction_text:
            self.primary_result_label.setText(
                f"{file_name} — "
                f"Whole Brain: {prediction_text}"
            )
        else:
            self.primary_result_label.setText(
                f"{file_name} — Whole Brain prediction "
                "has not been run."
            )

    def clear_files(self) -> None:
        """Remove loaded impacts and prediction results."""

        self.loaded_files.clear()
        self.impact_metadata.clear()
        self.impact_features.clear()
        self.impact_predictions.clear()
        self.impact_prediction_errors.clear()
        self.impact_statuses.clear()
        self.prediction_attempted_paths.clear()

        self.validation_table.setRowCount(0)
        self.prediction_table.setRowCount(0)

        self.status_label.setText(
            "No impact files loaded."
        )

        self.primary_result_label.setText(
            "Primary result: Whole Brain prediction "
            "has not been run."
        )

        self.update_button_state()

    def update_button_state(self) -> None:
        """Enable buttons only when their actions are available."""

        prediction_available = (
            self.predictor is not None
            and bool(self.predictor.models)
            and bool(self.impact_features)
        )

        export_available = (
            self.predictor is not None
            and bool(
                self.prediction_attempted_paths
            )
        )

        self.run_button.setEnabled(
            prediction_available
        )

        self.export_button.setEnabled(
            export_available
        )

    def update_status_label(self) -> None:
        """Display the impact validation summary."""

        total_impacts = len(
            self.loaded_files
        )

        valid_impacts = len(
            self.impact_features
        )

        invalid_impacts = (
            total_impacts - valid_impacts
        )

        self.status_label.setText(
            f"{total_impacts} impact file(s) loaded: "
            f"{valid_impacts} valid, "
            f"{invalid_impacts} invalid."
        )


def main() -> None:
    """Launch the desktop application."""

    application = QApplication(sys.argv)

    application.setApplicationName(
        "XGBoost Regional Brain Strain Predictor"
    )

    window = MainWindow()
    window.show()

    sys.exit(application.exec())


if __name__ == "__main__":
    main()
    