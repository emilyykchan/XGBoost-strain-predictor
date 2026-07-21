from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class MainWindow(QMainWindow):
    """Main window for the XGBoost strain prediction application."""

    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("XGBoost Strain Predictor")
        self.resize(1000, 600)

        # Keep a record of the full file paths.
        self.loaded_files: list[Path] = []

        title = QLabel("XGBoost Strain Predictor")
        title.setStyleSheet("font-size: 26px; font-weight: bold;")

        description = QLabel(
            "Batch prediction of regional brain strain "
            "from head-kinematics CSV files."
        )

        self.status_label = QLabel("No impact files loaded.")

        self.load_button = QPushButton("Load impact CSV files")
        self.load_button.setMinimumHeight(40)
        self.load_button.clicked.connect(self.load_csv_files)

        self.clear_button = QPushButton("Clear files")
        self.clear_button.setMinimumHeight(40)
        self.clear_button.clicked.connect(self.clear_files)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.clear_button)
        button_layout.addStretch()

        self.file_table = QTableWidget()
        self.file_table.setColumnCount(5)
        self.file_table.setHorizontalHeaderLabels(
            [
                "File",
                "Samples",
                "Columns",
                "Detected column names",
                "Status",
            ]
        )

        self.file_table.setAlternatingRowColors(True)
        self.file_table.setSortingEnabled(True)
        self.file_table.horizontalHeader().setStretchLastSection(True)

        layout = QVBoxLayout()
        layout.addWidget(title)
        layout.addWidget(description)
        layout.addSpacing(15)
        layout.addLayout(button_layout)
        layout.addWidget(self.status_label)
        layout.addWidget(self.file_table)

        central_widget = QWidget()
        central_widget.setLayout(layout)

        self.setCentralWidget(central_widget)

    def load_csv_files(self) -> None:
        """Open a file-selection window and load multiple CSV files."""

        selected_files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select head-kinematics CSV files",
            "",
            "CSV files (*.csv)",
        )

        if not selected_files:
            return

        self.file_table.setSortingEnabled(False)

        for selected_file in selected_files:
            file_path = Path(selected_file)

            # Prevent the same file being added more than once.
            if file_path in self.loaded_files:
                continue

            self.loaded_files.append(file_path)
            self.add_file_to_table(file_path)

        self.file_table.resizeColumnsToContents()
        self.file_table.setSortingEnabled(True)

        self.status_label.setText(
            f"{len(self.loaded_files)} impact file(s) loaded."
        )

    def add_file_to_table(self, file_path: Path) -> None:
        """Read one CSV and display basic information about it."""

        row_number = self.file_table.rowCount()
        self.file_table.insertRow(row_number)

        file_name = file_path.name
        sample_count = ""
        column_count = ""
        detected_columns = ""
        status = "Valid CSV"

        try:
            dataframe = pd.read_csv(file_path)

            sample_count = str(len(dataframe))
            column_count = str(len(dataframe.columns))
            detected_columns = ", ".join(
                str(column) for column in dataframe.columns
            )

            if dataframe.empty:
                status = "Invalid: empty CSV"

            elif len(dataframe.columns) == 0:
                status = "Invalid: no columns"

            elif len(dataframe) < 3:
                status = "Warning: fewer than 3 samples"

        except pd.errors.EmptyDataError:
            status = "Invalid: empty CSV"

        except pd.errors.ParserError:
            status = "Invalid: CSV could not be parsed"

        except UnicodeDecodeError:
            status = "Invalid: unsupported text encoding"

        except OSError as error:
            status = f"Invalid: {error}"

        values = [
            file_name,
            sample_count,
            column_count,
            detected_columns,
            status,
        ]

        for column_number, value in enumerate(values):
            table_item = QTableWidgetItem(value)

            # Prevent the user accidentally editing table values.
            table_item.setFlags(
                table_item.flags() & ~Qt.ItemFlag.ItemIsEditable
            )

            self.file_table.setItem(
                row_number,
                column_number,
                table_item,
            )

    def clear_files(self) -> None:
        """Remove all loaded files from the application."""

        self.loaded_files.clear()
        self.file_table.setRowCount(0)
        self.status_label.setText("No impact files loaded.")


def main() -> None:
    """Launch the desktop application."""

    application = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(application.exec())


if __name__ == "__main__":
    main()