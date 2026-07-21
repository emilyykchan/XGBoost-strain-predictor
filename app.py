from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from kinematics import extract_impact_features


class MainWindow(QMainWindow):
    """Main window for the XGBoost strain prediction application."""

    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("XGBoost Strain Predictor")
        self.resize(1450, 700)

        # Stores the full paths of CSV files currently loaded.
        self.loaded_files: list[Path] = []

        # Main title
        title_label = QLabel("XGBoost Strain Predictor")
        title_label.setStyleSheet(
            "font-size: 26px; font-weight: bold;"
        )

        # Description of the required input structure
        description_label = QLabel(
            "Each CSV represents one head impact. "
            "Each row represents one time point in the impact time history."
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

        self.status_label = QLabel("No impact files loaded.")

        # Buttons
        self.load_button = QPushButton("Load impact CSV files")
        self.load_button.setMinimumHeight(40)
        self.load_button.clicked.connect(
            self.load_csv_files
        )

        self.clear_button = QPushButton("Clear files")
        self.clear_button.setMinimumHeight(40)
        self.clear_button.clicked.connect(
            self.clear_files
        )

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.clear_button)
        button_layout.addStretch()

        # Main results table
        self.file_table = QTableWidget()
        self.file_table.setColumnCount(9)

        self.file_table.setHorizontalHeaderLabels(
            [
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
        )

        self.file_table.setAlternatingRowColors(True)
        self.file_table.setSortingEnabled(True)
        self.file_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.file_table.setSelectionMode(
            QTableWidget.SelectionMode.SingleSelection
        )
        self.file_table.setWordWrap(False)

        # Column resizing behaviour
        header = self.file_table.horizontalHeader()
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

        self.file_table.setColumnWidth(2, 320)
        self.file_table.setColumnWidth(3, 420)

        # Main window layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(title_label)
        main_layout.addWidget(description_label)
        main_layout.addWidget(required_columns_label)
        main_layout.addWidget(units_label)
        main_layout.addSpacing(15)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.status_label)
        main_layout.addWidget(self.file_table)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)

        self.setCentralWidget(central_widget)

    def load_csv_files(self) -> None:
        """Select and process multiple head-impact CSV files."""

        selected_files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select head-impact CSV files",
            "",
            "CSV files (*.csv)",
        )

        if not selected_files:
            return

        self.file_table.setSortingEnabled(False)

        added_count = 0

        for selected_file in selected_files:
            file_path = Path(selected_file).resolve()

            # Do not add the same file twice.
            if file_path in self.loaded_files:
                continue

            self.loaded_files.append(file_path)
            self.add_impact_to_table(file_path)
            added_count += 1

        self.file_table.setSortingEnabled(True)

        if added_count == 0:
            self.status_label.setText(
                "No new files were added. "
                "The selected files were already loaded."
            )
        else:
            self.update_status_label()

    def add_impact_to_table(
        self,
        file_path: Path,
    ) -> None:
        """
        Extract the four XGBoost input features from one impact.

        One CSV produces one table row.
        """

        row_number = self.file_table.rowCount()
        self.file_table.insertRow(row_number)

        # Default values, used if validation fails.
        values = [
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

        try:
            result = extract_impact_features(file_path)
            feature_values = result["feature_values"]

            values = [
                result["file_name"],
                str(result["time_points"]),
                result["detected_columns"],
                result["interpretation"],
                f"{feature_values[0]:.6f}",
                f"{feature_values[1]:.6f}",
                f"{feature_values[2]:.6f}",
                f"{feature_values[3]:.6f}",
                "Ready for prediction",
            ]

        except (
            ValueError,
            OSError,
            UnicodeDecodeError,
        ) as error:
            values[-1] = f"Invalid: {error}"

        for column_number, value in enumerate(values):
            table_item = QTableWidgetItem(str(value))

            # Prevent users from editing calculated values.
            table_item.setFlags(
                table_item.flags()
                & ~Qt.ItemFlag.ItemIsEditable
            )

            # Show the full text when hovering over a cell.
            table_item.setToolTip(str(value))

            self.file_table.setItem(
                row_number,
                column_number,
                table_item,
            )

    def clear_files(self) -> None:
        """Remove all loaded impact files and table results."""

        self.loaded_files.clear()
        self.file_table.setRowCount(0)
        self.status_label.setText(
            "No impact files loaded."
        )

    def update_status_label(self) -> None:
        """Update the impact and validation summary."""

        total_impacts = len(self.loaded_files)
        valid_impacts = 0
        invalid_impacts = 0

        status_column = 8

        for row_number in range(
            self.file_table.rowCount()
        ):
            status_item = self.file_table.item(
                row_number,
                status_column,
            )

            if status_item is None:
                continue

            if status_item.text() == "Ready for prediction":
                valid_impacts += 1
            else:
                invalid_impacts += 1

        self.status_label.setText(
            f"{total_impacts} impact file(s) loaded: "
            f"{valid_impacts} valid, "
            f"{invalid_impacts} invalid."
        )


def main() -> None:
    """Launch the desktop application."""

    application = QApplication(sys.argv)

    application.setApplicationName(
        "XGBoost Strain Predictor"
    )

    window = MainWindow()
    window.show()

    sys.exit(application.exec())


if __name__ == "__main__":
    main()
    