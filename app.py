from __future__ import annotations

import csv
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
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


APP_STYLESHEET = """
QMainWindow {
    background-color: #f3f6fa;
}

QWidget {
    color: #1f2937;
    font-size: 13px;
}

QFrame#Card {
    background-color: #ffffff;
    border: 1px solid #dce4ed;
    border-radius: 10px;
}

QLabel#TitleLabel {
    color: #14213d;
    font-size: 28px;
    font-weight: 700;
}

QLabel#StatusLabel {
    color: #475569;
    font-weight: 600;
    padding: 2px 4px;
}

QLabel#AttributionLabel {
    color: #64748b;
    font-size: 11px;
    padding: 2px 4px;
}

QPushButton {
    background-color: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 7px;
    color: #263548;
    font-weight: 600;
    padding: 8px 16px;
}

QPushButton:hover {
    background-color: #f1f5f9;
    border-color: #94a3b8;
}

QPushButton:pressed {
    background-color: #e2e8f0;
}

QPushButton:disabled {
    background-color: #e9eef4;
    border-color: #d8e0e8;
    color: #94a3b8;
}

QPushButton#PrimaryButton {
    background-color: #2e7d32;
    border-color: #2e7d32;
    color: #ffffff;
}

QPushButton#PrimaryButton:hover {
    background-color: #388e3c;
    border-color: #388e3c;
}

QPushButton#PrimaryButton:pressed {
    background-color: #1b5e20;
    border-color: #1b5e20;
}

QPushButton#PrimaryButton:disabled {
    background-color: #b7d8ba;
    border-color: #b7d8ba;
    color: #f7fbf7;
}

QPushButton#ExportButton {
    background-color: #2563eb;
    border-color: #2563eb;
    color: #ffffff;
}

QPushButton#ExportButton:hover {
    background-color: #3474f0;
    border-color: #3474f0;
}

QPushButton#ExportButton:pressed {
    background-color: #1d4ed8;
    border-color: #1d4ed8;
}

QPushButton#ExportButton:disabled {
    background-color: #b8c9ed;
    border-color: #b8c9ed;
    color: #f7f9ff;
}

QPushButton#ClearButton {
    background-color: #ffffff;
    border-color: #e5aaa5;
    color: #b42318;
}

QPushButton#ClearButton:hover {
    background-color: #fff3f2;
    border-color: #d97068;
}

QPushButton#ClearButton:pressed {
    background-color: #fee4e2;
}

QTabWidget::pane {
    background-color: #ffffff;
    border: 1px solid #dce4ed;
    border-radius: 8px;
    top: -1px;
}

QTabBar::tab {
    background-color: #e8edf3;
    border: 1px solid #d5dde6;
    border-bottom: none;
    border-top-left-radius: 7px;
    border-top-right-radius: 7px;
    color: #526174;
    font-weight: 600;
    min-width: 150px;
    padding: 9px 18px;
    margin-right: 3px;
}

QTabBar::tab:hover {
    background-color: #f2f5f8;
}

QTabBar::tab:selected {
    background-color: #ffffff;
    color: #1d3557;
}

QTableWidget {
    background-color: #ffffff;
    alternate-background-color: #f8fafc;
    border: none;
    gridline-color: #e5eaf0;
    selection-background-color: #dbeafe;
    selection-color: #172033;
}

QHeaderView::section {
    background-color: #edf2f7;
    border: none;
    border-bottom: 1px solid #d6dee7;
    border-right: 1px solid #d6dee7;
    color: #334155;
    font-weight: 700;
    padding: 8px;
}

QTableCornerButton::section {
    background-color: #edf2f7;
    border: none;
}

QToolTip {
    background-color: #172033;
    border: 1px solid #172033;
    color: #ffffff;
    padding: 5px;
}
"""


def resource_path(relative_path: str) -> Path:
    """
    Return a resource path that works during development and
    inside a packaged PyInstaller application.
    """

    bundled_directory = getattr(
        sys,
        "_MEIPASS",
        None,
    )

    if bundled_directory is not None:
        base_directory = Path(bundled_directory)
    else:
        base_directory = Path(__file__).resolve().parent

    return base_directory / relative_path


class MainWindow(QMainWindow):
    """Main window for the brain strain prediction application."""

    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle(
            "XGBoost Brain Strain Predictor"
        )
        self.resize(1500, 760)

        self.loaded_files: list[Path] = []

        self.impact_metadata: dict[
            Path,
            dict[str, object],
        ] = {}

        self.impact_features: dict[
            Path,
            object,
        ] = {}

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

        self.prediction_attempted_paths: set[Path] = set()

        self.predictor: RegionalPredictor | None = None
        self.predictor_error: str | None = None

        try:
            self.predictor = RegionalPredictor()

        except ModelConfigurationError as error:
            self.predictor_error = str(error)

        self.region_column_by_id: dict[str, int] = {}

        # --------------------------------------------------------------
        # Header
        # --------------------------------------------------------------

        title_label = QLabel(
            "XGBoost Brain Strain Predictor"
        )
        title_label.setObjectName("TitleLabel")

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
            "font-weight: 700;"
        )
        required_columns_label.setWordWrap(True)

        units_label = QLabel(
            "Required units: rotational velocity in rad/s and "
            "rotational acceleration in rad/s²."
        )
        units_label.setWordWrap(True)

        header_card = QFrame()
        header_card.setObjectName("Card")

        header_layout = QVBoxLayout(header_card)
        header_layout.setContentsMargins(
            18,
            15,
            18,
            15,
        )
        header_layout.setSpacing(5)
        header_layout.addWidget(title_label)
        header_layout.addWidget(description_label)
        header_layout.addWidget(required_columns_label)
        header_layout.addWidget(units_label)

        # --------------------------------------------------------------
        # Buttons
        # --------------------------------------------------------------

        self.load_button = QPushButton(
            "Load impact CSV files"
        )
        self.load_button.setMinimumHeight(42)
        self.load_button.clicked.connect(
            self.load_csv_files
        )

        self.run_button = QPushButton(
            "Run strain prediction"
        )
        self.run_button.setObjectName(
            "PrimaryButton"
        )
        self.run_button.setMinimumHeight(42)
        self.run_button.clicked.connect(
            self.run_predictions
        )

        self.export_button = QPushButton(
            "Export strain predictions to CSV"
        )
        self.export_button.setObjectName(
            "ExportButton"
        )
        self.export_button.setMinimumHeight(42)
        self.export_button.clicked.connect(
            self.export_predictions_csv
        )

        self.clear_button = QPushButton(
            "Clear files"
        )
        self.clear_button.setObjectName(
            "ClearButton"
        )
        self.clear_button.setMinimumHeight(42)
        self.clear_button.clicked.connect(
            self.clear_files
        )

        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(
            12,
            10,
            12,
            10,
        )
        button_layout.setSpacing(9)

        button_layout.addWidget(
            self.load_button
        )
        button_layout.addWidget(
            self.run_button
        )
        button_layout.addWidget(
            self.export_button
        )
        button_layout.addWidget(
            self.clear_button
        )
        button_layout.addStretch()

        controls_card = QFrame()
        controls_card.setObjectName("Card")
        controls_card.setLayout(button_layout)

        # --------------------------------------------------------------
        # Status
        # --------------------------------------------------------------

        self.status_label = QLabel(
            "No impact files loaded."
        )
        self.status_label.setObjectName(
            "StatusLabel"
        )

        # --------------------------------------------------------------
        # Tables and tabs
        # --------------------------------------------------------------

        self.validation_table = QTableWidget()
        self.configure_validation_table()

        self.prediction_table = QTableWidget()
        self.configure_prediction_table()

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        self.tabs.addTab(
            self.validation_table,
            "Input validation",
        )

        self.tabs.addTab(
            self.prediction_table,
            "Strain predictions",
        )

        # --------------------------------------------------------------
        # Attribution
        # --------------------------------------------------------------

        attribution_label = QLabel(
            "Part of TRACE. Developed by the HEAD Lab, "
            "Imperial College London."
        )
        attribution_label.setObjectName(
            "AttributionLabel"
        )
        attribution_label.setAlignment(
            Qt.AlignmentFlag.AlignRight
            | Qt.AlignmentFlag.AlignVCenter
        )

        # --------------------------------------------------------------
        # Main layout
        # --------------------------------------------------------------

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(
            16,
            16,
            16,
            12,
        )
        main_layout.setSpacing(11)

        main_layout.addWidget(header_card)
        main_layout.addWidget(controls_card)
        main_layout.addWidget(
            self.status_label
        )
        main_layout.addWidget(
            self.tabs,
            stretch=1,
        )
        main_layout.addWidget(
            attribution_label
        )

        central_widget = QWidget()
        central_widget.setLayout(
            main_layout
        )

        self.setCentralWidget(
            central_widget
        )

        self.update_button_state()

    def configure_validation_table(self) -> None:
        """Configure the input-validation table."""

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
            self.validation_table
            .horizontalHeader()
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
        """Configure the strain-prediction table."""

        loaded_specs = []

        if self.predictor is not None:
            loaded_specs = (
                self.predictor.loaded_specs
            )

        headers = ["Impact file"]

        for column_number, spec in enumerate(
            loaded_specs,
            start=1,
        ):
            headers.append(
                spec.display_name
            )

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
            self.prediction_table
            .horizontalHeader()
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
            primary_spec = (
                self.predictor.primary_spec
            )

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
        """Apply common display settings to a table."""

        table.setAlternatingRowColors(True)
        table.setSortingEnabled(True)
        table.setWordWrap(False)
        table.setShowGrid(False)
        table.setCornerButtonEnabled(False)

        table.verticalHeader().setVisible(
            False
        )

        table.verticalHeader().setDefaultSectionSize(
            34
        )

        table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )

        table.setSelectionMode(
            QTableWidget.SelectionMode.SingleSelection
        )

    def load_csv_files(self) -> None:
        """Select and validate multiple impact CSV files."""

        selected_files, _ = (
            QFileDialog.getOpenFileNames(
                self,
                "Select head-impact CSV files",
                "",
                "CSV files (*.csv)",
            )
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

            self.loaded_files.append(
                file_path
            )

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
                self.prediction_table
                .columnCount()
            )
        ]

        prediction_values[0] = (
            file_path.name
        )

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

            status_text = (
                "Ready for prediction"
            )

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
            error_text = (
                f"Invalid: {error}"
            )

            self.impact_statuses[
                file_path
            ] = error_text

            validation_values[-1] = (
                error_text
            )

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
            item = (
                self.create_read_only_item(
                    value
                )
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
        """Predict strain for all available models."""

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
                "No strain prediction models "
                "could be loaded.",
            )
            return

        if not self.impact_features:
            QMessageBox.information(
                self,
                "No valid impacts",
                "Load at least one valid "
                "impact CSV first.",
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
            file_item = (
                self.prediction_table.item(
                    row_number,
                    0,
                )
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
                status_text = (
                    "Prediction complete"
                )
                successful_impacts += 1

            elif predictions and errors:
                status_text = (
                    "Partial prediction: "
                    f"{len(errors)} model(s) failed"
                )
                partial_impacts += 1

            else:
                status_text = (
                    "Prediction failed"
                )
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

        # Automatically open the results tab.
        self.tabs.setCurrentWidget(
            self.prediction_table
        )

        self.update_button_state()

    def export_predictions_csv(self) -> None:
        """Export filenames, time points and predictions."""

        if self.predictor is None:
            QMessageBox.critical(
                self,
                "Models unavailable",
                "The model configuration "
                "is unavailable.",
            )
            return

        if not self.prediction_attempted_paths:
            QMessageBox.information(
                self,
                "No predictions",
                "Run strain prediction "
                "before exporting.",
            )
            return

        selected_path, _ = (
            QFileDialog.getSaveFileName(
                self,
                "Export strain predictions to CSV",
                "strain_predictions.csv",
                "CSV files (*.csv)",
            )
        )

        if not selected_path:
            return

        output_path = Path(
            selected_path
        )

        if output_path.suffix.lower() != ".csv":
            output_path = (
                output_path.with_suffix(
                    ".csv"
                )
            )

        loaded_specs = (
            self.predictor.loaded_specs
        )

        headers = [
            "Impact file",
            "Time points",
            *[
                spec.display_name
                for spec in loaded_specs
            ],
        ]

        try:
            with output_path.open(
                "w",
                newline="",
                encoding="utf-8-sig",
            ) as output_file:
                writer = csv.writer(
                    output_file
                )

                writer.writerow(headers)

                for file_path in self.loaded_files:
                    metadata = (
                        self.impact_metadata.get(
                            file_path
                        )
                    )

                    if metadata is None:
                        time_points = ""
                    else:
                        time_points = (
                            metadata.get(
                                "time_points",
                                "",
                            )
                        )

                    predictions = (
                        self.impact_predictions.get(
                            file_path,
                            {},
                        )
                    )

                    prediction_cells = [
                        (
                            f"{predictions[spec.model_id]:.10f}"
                            if spec.model_id
                            in predictions
                            else ""
                        )
                        for spec in loaded_specs
                    ]

                    writer.writerow(
                        [
                            file_path.name,
                            time_points,
                            *prediction_cells,
                        ]
                    )

        except OSError as error:
            QMessageBox.critical(
                self,
                "Export failed",
                "The CSV could not be saved:\n"
                f"{error}",
            )
            return

        QMessageBox.information(
            self,
            "Export complete",
            "Predictions were saved to:\n"
            f"{output_path}",
        )

    def clear_files(self) -> None:
        """Remove all loaded impacts and predictions."""

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

        self.tabs.setCurrentWidget(
            self.validation_table
        )

        self.update_button_state()

    def update_button_state(self) -> None:
        """Enable buttons only when actions are available."""

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
        """Display the impact-validation summary."""

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
        "XGBoost Brain Strain Predictor"
    )

    application.setStyle("Fusion")
    application.setStyleSheet(
        APP_STYLESHEET
    )

    icon_path = resource_path(
        "docs/assets/trace-mark.svg"
    )

    if icon_path.exists():
        application.setWindowIcon(
            QIcon(str(icon_path))
        )

    window = MainWindow()

    if icon_path.exists():
        window.setWindowIcon(
            QIcon(str(icon_path))
        )

    window.show()

    sys.exit(
        application.exec()
    )


if __name__ == "__main__":
    main()