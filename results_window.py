from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QLabel, QWidget, 
                           QTableWidget, QTableWidgetItem, QHeaderView, QFrame)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class ResultsWindow(QDialog):
    STYLE_SHEET = """
        QDialog {
            background-color: white;
        }
        QLabel {
            font-size: 14px;
            margin: 10px;
            color: #2c3e50;
        }
        QTableWidget {
            border: none;
            gridline-color: #ecf0f1;
            font-size: 13px;
            color: black;
        }
        QTableWidget::item {
            padding: 5px;
            color: black;
        }
        QHeaderView::section {
            background-color: #f0f0f0;
            color: black;
            padding: 8px;
            font-size: 13px;
            border: 1px solid #d3d3c7;
            font-weight: bold;
        }
        QTableWidget::item:selected {
            background-color: #e8f6ff;
            color: black;
        }
    """

    def __init__(self, results, parent=None):
        super().__init__(parent)
        self.results = results
        self.is_reaction_mode = results[0]['user_input'] == 'SPACE' if results else False
        self.setWindowTitle("Wyniki Testu")
        self.setGeometry(200, 200, 800, 800)
        self.setStyleSheet(self.STYLE_SHEET)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        summary_frame = QFrame()
        summary_frame.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border-radius: 10px;
                padding: 15px;
            }
            QLabel {
                font-size: 15px;
                color: #2c3e50;
            }
        """)
        summary_layout = QVBoxLayout(summary_frame)
        
        total_tests = len(self.results)
        avg_time = sum(r['reaction_time'] for r in self.results) / total_tests if total_tests > 0 else 0
        
        if self.is_reaction_mode:
            summary = QLabel(
                f"Podsumowanie testu czasu reakcji:\n"
                f"Liczba testów: {total_tests}\n"
                f"Średni czas reakcji: {avg_time:.2f}s"
            )
        else:
            correct_answers = sum(1 for r in self.results if r['correct'])
            accuracy = (correct_answers / total_tests) * 100 if total_tests > 0 else 0
            summary = QLabel(
                f"Podsumowanie testu rozpoznawania liczb:\n"
                f"Liczba testów: {total_tests}\n"
                f"Poprawne odpowiedzi: {correct_answers}\n"
                f"Dokładność: {accuracy:.1f}%\n"
                f"Średni czas reakcji: {avg_time:.2f}s"
            )
        
        summary.setStyleSheet("QLabel { font-size: 14px; margin: 10px; }")
        summary_layout.addWidget(summary)
        
        layout.addWidget(summary_frame)

        table = QTableWidget()
        if self.is_reaction_mode:
            table.setColumnCount(3)
            table.setHorizontalHeaderLabels([
                "Nr testu", 
                "Czas reakcji (ms)", 
                "Intensywność składowej koloru"
            ])
        else:
            table.setColumnCount(4)
            table.setHorizontalHeaderLabels([
                "Nr testu", 
                "Czas reakcji (ms)", 
                "Intensywność składowej koloru",
                "Poprawność"
            ])
        
        table.setRowCount(len(self.results))
        for i, result in enumerate(self.results):
            table.setItem(i, 0, QTableWidgetItem(str(result['test_number'])))
            ms_time = int(result['reaction_time'] * 1000)
            table.setItem(i, 1, QTableWidgetItem(str(ms_time)))
            table.setItem(i, 2, QTableWidgetItem(str(result['intensity'])))
            if not self.is_reaction_mode:
                table.setItem(i, 3, QTableWidgetItem("Tak" if result['correct'] else "Nie"))

        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        table.setStyleSheet("""
            QTableWidget {
                gridline-color: #d3d3d3;
                font-size: 12px;
            }
            QHeaderView::section {
                background-color: #f0f0f0;
                padding: 4px;
                font-size: 12px;
                border: 1px solid #d3d3d3;
            }
        """)
        
        layout.addWidget(table)

        self.figure = Figure(figsize=(8, 4))
        self.canvas = FigureCanvas(self.figure)
        self._create_intensity_chart()
        layout.addWidget(self.canvas)

        self.setLayout(layout)

    def _create_intensity_chart(self):
        ax = self.figure.add_subplot(111)
        test_numbers = [r['test_number'] for r in self.results]
        intensities = [r['intensity'] for r in self.results]
        colors = [{'R': 'red', 'G': 'green', 'B': 'blue'}[r['color_component']] 
                 for r in self.results]
        
        ax.bar(test_numbers, intensities, color=colors)
        ax.set_xlabel('Numer testu')
        ax.set_ylabel('Próg intensywności')
        ax.set_title('Progi intensywności w kolejnych testach')
        
        self.figure.tight_layout() 