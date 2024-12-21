import sys
import csv
import time
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QPushButton, QLabel, QFileDialog, QProgressBar,
                           QMessageBox, QHBoxLayout, QInputDialog,
                           QLineEdit, QRadioButton, QButtonGroup, QFrame)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage, qRgb
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import cv2
import os
from typing import List, Dict, Any, Optional
import numpy.typing as npt
from image_numbers import IMAGE_NUMBERS
from results_window import ResultsWindow
from shutil import copy2

class ColorPerceptionTest(QMainWindow):
    """
    Aplikacja do testowania percepcji barw.
    
    Wyświetla serię obrazów testowych i mierzy czas reakcji użytkownika
    na rozpoznanie ukrytych liczb przy różnych poziomach intensywności
    składowych kolorów RGB.
    """

    WINDOW_TITLE = "Test Percepcji Barw"
    WINDOW_GEOMETRY = (100, 100, 800, 600)
    IMAGE_SIZE = (400, 400)
    TIMER_INTERVAL = 40  # ms
    INTENSITY_STEP = 3
    MAX_TEST_TIME = 15  # seconds
    COLOR_COMPONENTS = ['R', 'G', 'B']
    DEFAULT_MAX_TESTS = 4  # Liczba poczatkowych obrazkow
    
    STYLE_SHEET = """
        QMainWindow {
            background-color: #f0f0f0;
        }
        QLabel {
            font-size: 14px;
            margin: 10px;
            color: #2c3e50;
        }
        QPushButton {
            padding: 10px 20px;
            font-size: 14px;
            border-radius: 5px;
            border: none;
            color: white;
            background-color: #3498db;
        }
        QPushButton:hover {
            background-color: #2980b9;
        }
        QPushButton:disabled {
            background-color: #bdc3c7;
        }
        QProgressBar {
            border: 2px solid #bdc3c7;
            border-radius: 5px;
            text-align: center;
            height: 25px;
        }
        QProgressBar::chunk {
            background-color: #2ecc71;
            border-radius: 3px;
        }
        QLineEdit {
            padding: 8px;
            font-size: 14px;
            border: 2px solid #bdc3c7;
            border-radius: 5px;
            background-color: white;
        }
        QLineEdit:focus {
            border-color: #3498db;
        }
        QRadioButton {
            font-size: 14px;
            color: #2c3e50;
            spacing: 8px;
        }
        QRadioButton::indicator {
            width: 18px;
            height: 18px;
        }
        QRadioButton::indicator:checked {
            background-color: #3498db;
            border: 2px solid #2980b9;
            border-radius: 9px;
        }
        QRadioButton::indicator:unchecked {
            background-color: white;
            border: 2px solid #bdc3c7;
            border-radius: 9px;
        }
    """
    
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(self.WINDOW_TITLE)
        self.setGeometry(*self.WINDOW_GEOMETRY)
        self.setStyleSheet(self.STYLE_SHEET)
        
        self.current_test: int = 0
        self.max_tests: int = self.DEFAULT_MAX_TESTS  
        self.start_time: Optional[float] = None
        self.results: List[Dict[str, Any]] = []
        self.current_intensity: float = 0
        self.color_component: str = 'R'
        self.test_running: bool = False
        self.test_images: List[npt.NDArray] = []
        self.current_image = None
        
        self.load_test_images()
        self.init_ui()
        
    def init_ui(self):
        central_widget = QWidget()
        central_widget.setStyleSheet("background-color: white; border-radius: 10px;")
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        mode_layout = QHBoxLayout()
        mode_label = QLabel("Wybierz tryb testu:")
        mode_label.setStyleSheet("font-weight: bold;")
        mode_layout.addWidget(mode_label)
        
        self.mode_group = QButtonGroup()
        self.number_mode = QRadioButton("Test rozpoznawania liczb")
        self.reaction_mode = QRadioButton("Test czasu reakcji")
        self.number_mode.setChecked(True)  # Default mode
        
        self.mode_group.addButton(self.number_mode)
        self.mode_group.addButton(self.reaction_mode)
        
        mode_layout.addWidget(self.number_mode)
        mode_layout.addWidget(self.reaction_mode)
        layout.addLayout(mode_layout)
        
        intensity_layout = QHBoxLayout()
        intensity_label = QLabel("Prędkość zmiany progu składowej koloru (domyślnie 3):")
        intensity_label.setStyleSheet("font-weight: bold;")
        self.intensity_input = QLineEdit()
        self.intensity_input.setPlaceholderText("3")
        self.intensity_input.setMaximumWidth(100)
        self.intensity_input.setStyleSheet("""
            QLineEdit {
                padding: 5px;
                font-size: 14px;
                border: 2px solid #bdc3c7;
                border-radius: 5px;
            }
        """)
        
        intensity_layout.addWidget(intensity_label)
        intensity_layout.addWidget(self.intensity_input)
        intensity_layout.addStretch()  
        layout.addLayout(intensity_layout)
        
        layout.addSpacing(10)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        
        image_frame = QFrame()
        image_frame.setStyleSheet("""
            QFrame {
                border: 2px solid #bdc3c7;
                border-radius: 10px;
                background-color: #f8f9fa;
                padding: 10px;
            }
        """)
        image_layout = QVBoxLayout(image_frame)
        image_layout.addWidget(self.image_label)
        layout.addWidget(image_frame)
        
        self.input_layout = QHBoxLayout()
        input_label = QLabel("Wprowadź widoczną liczbę:")
        self.number_input = QLineEdit()
        self.number_input.setEnabled(False)
        self.number_input.returnPressed.connect(self.handle_number_input)
        self.number_input.setStyleSheet("QLineEdit { padding: 5px; font-size: 14px; }")
        self.input_layout.addWidget(input_label)
        self.input_layout.addWidget(self.number_input)
        layout.addLayout(self.input_layout)
        
        progress_layout = QHBoxLayout()
        progress_label = QLabel("Postęp testu:")
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(progress_label)
        progress_layout.addWidget(self.progress_bar)
        layout.addLayout(progress_layout)
        
        button_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start testu")
        self.start_button.clicked.connect(self.start_test)
        
        self.stop_button = QPushButton("Przerwij test")
        self.stop_button.clicked.connect(self.handle_stop_resume)
        self.stop_button.setEnabled(False)
        
        self.add_image_button = QPushButton("Dodaj własny obraz")
        self.add_image_button.clicked.connect(self.add_custom_image)
        
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71;
                padding: 12px 25px;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
        """)
        
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                padding: 12px 25px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        
        self.add_image_button.setStyleSheet("""
            QPushButton {
                background-color: #9b59b6;
                padding: 12px 25px;
            }
            QPushButton:hover {
                background-color: #8e44ad;
            }
        """)
        
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.add_image_button)
        layout.addLayout(button_layout)
        
        instructions_layout = QVBoxLayout()
        
        self.status_label = QLabel("Naciśnij 'Start testu' aby rozpocząć")
        self.status_label.setStyleSheet("""
            QLabel {
                font-size: 15px;
                color: #2c3e50;
                font-weight: bold;
                padding: 10px;
                background-color: #ecf0f1;
                border-radius: 5px;
            }
        """)
        instructions_layout.addWidget(self.status_label)
        
        self.shortcuts_label = QLabel()
        self.update_shortcuts_label()
        self.shortcuts_label.setAlignment(Qt.AlignCenter)
        self.shortcuts_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                color: #7f8c8d;
                font-style: italic;
            }
        """)
        instructions_layout.addWidget(self.shortcuts_label)
        
        layout.addLayout(instructions_layout)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_color_intensity)
        
        self.figure = Figure(figsize=(6, 4))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        self.mode_group.buttonClicked.connect(self.on_mode_changed)

    def on_mode_changed(self):
        """Handle test mode change"""
        is_number_mode = self.number_mode.isChecked()
        self.number_input.setVisible(is_number_mode)
        self.input_layout.itemAt(0).widget().setVisible(is_number_mode) 
        self.update_shortcuts_label()

    def update_shortcuts_label(self):
        """Update instructions based on selected mode"""
        if self.number_mode.isChecked():
            self.shortcuts_label.setText("Naciśnij Enter gdy wprowadzisz liczbę rozpoznaną na obrazie")
        else:
            self.shortcuts_label.setText("Naciśnij SPACJĘ gdy rozpoznasz liczbę na obrazie")

    def load_test_images(self) -> None:
        """Load test images from the test_images directory."""
        self.test_images = [] 
        try:
            self._load_images_from_directory()
            
            if not self.test_images:
                self._reset_image_numbers()
           
            self.max_tests = len(self.test_images)
        except Exception as e:
            print(f"Warning: Could not load test images: {str(e)}")

    def _load_images_from_directory(self) -> None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        image_dir = os.path.join(current_dir, 'test_images')
        
        if not os.path.exists(image_dir):
            print(f"Warning: Directory {image_dir} does not exist")
            return
        
        for i in range(1, self.DEFAULT_MAX_TESTS + 1):
            self._load_single_image(image_dir, i)
        
        custom_images = [f for f in os.listdir(image_dir) 
                        if f.startswith('it-') and 
                        int(f.split('-')[1].split('.')[0]) > self.DEFAULT_MAX_TESTS]
        
        for image_file in sorted(custom_images, 
                               key=lambda x: int(x.split('-')[1].split('.')[0])):
            img_num = int(image_file.split('-')[1].split('.')[0])
            self._load_single_image(image_dir, img_num)

    def _load_single_image(self, image_dir: str, index: int) -> None:
        """
        Ładuje pojedynczy obraz testowy.
        
        Args:
            image_dir: Ścieżka do katalogu z obrazami
            index: Numer obrazu do załadowania
        """
        image_path = os.path.join(image_dir, f'it-{index}.png')
        
        if os.path.exists(image_path):
            try:
                img = cv2.imread(image_path)
                if img is not None:
                    # Konwersja z BGR do RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    self.test_images.append(img)
                    print(f"Pomyślnie załadowano obraz: it-{index}.png")
                else:
                    print(f"Nie udao się załadować obrazu: it-{index}.png")
            except Exception as e:
                print(f"Błąd podczas ładowania obrazu it-{index}.png: {str(e)}")
        else:
            print(f"Nie znaleziono pliku: {image_path}")

    def start_test(self):
        if not self.test_images:
            QMessageBox.warning(
                self, 
                "Ostrzeżenie", 
                "Test nie może zostać rozpoczęty, brak obrazów testowych."
            )
            return
        
        custom_step = self.intensity_input.text().strip()
        if custom_step:
            try:
                step = float(custom_step)
                if 0 < step <= 10:  
                    self.INTENSITY_STEP = step
                else:
                    QMessageBox.warning(
                        self, 
                        "Ostrzeżenie", 
                        "Krok intensywności musi być między 0 a 10.\nUżywam wartości domyślnej (3)."
                    )
                    self.INTENSITY_STEP = 3
            except ValueError:
                QMessageBox.warning(
                    self, 
                    "Ostrzeżenie", 
                    "Nieprawidłowa wartość kroku intensywności.\nUżywam wartości domyślnej (3)."
                )
                self.INTENSITY_STEP = 3
        else:
            self.INTENSITY_STEP = 3
        
        self.intensity_input.setEnabled(False)
        
        self.test_running = True
        self.current_test = 0
        self.results = []
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setValue(0)
        
        self.start_button.setText("Restart testu")
        self.stop_button.setText("Przerwij test")
        
        self.number_mode.setEnabled(False)
        self.reaction_mode.setEnabled(False)
        
        if self.number_mode.isChecked():
            self.number_input.setEnabled(True)
            self.number_input.setFocus()
        else:
            self.number_input.setEnabled(False)
            self.setFocus()
        
        self.load_next_test()

    def stop_test(self):
        self.test_running = False
        self.timer.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(True)  
        self.number_input.setEnabled(False)
        self.intensity_input.setEnabled(True)
        
        self.stop_button.setText("Wznów test")
        
        self.status_label.setText("Test zatrzymany")

    def finish_test(self):
        self.test_running = False
        self.timer.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.number_input.setEnabled(False)
        self.intensity_input.setEnabled(True)
        
        self.start_button.setText("Restart testu")
        self.start_button.clicked.disconnect() 
        self.start_button.clicked.connect(self.restart_test)
        
        self.number_mode.setEnabled(True)
        self.reaction_mode.setEnabled(True)
        
        self.status_label.setText("Test zakończony - wyświetlam wyniki")
        
        self.save_results()
        results_window = ResultsWindow(self.results, self)
        results_window.exec_()

    def restart_test(self):
        """Reset the test to initial state without starting it"""
        self.current_test = 0
        self.results = []
        self.current_intensity = 0
        self.progress_bar.setValue(0)
        self.intensity_input.setEnabled(True)
        
        self.start_button.setText("Start testu")
        self.start_button.clicked.disconnect()
        self.start_button.clicked.connect(self.start_test)
        self.stop_button.setText("Przerwij test")
        
        self.status_label.setText("Naciśnij 'Start testu' aby rozpocząć")
        
        self.image_label.clear()

    def save_results(self):
        results_dir = 'results'
        if not os.path.exists(results_dir):
            try:
                os.makedirs(results_dir)
            except Exception as e:
                QMessageBox.critical(self, "Błąd", 
                                   f"Nie można utworzyć katalogu results: {str(e)}")
                return

        filename = f"wyniki_testu_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        path = os.path.join(results_dir, filename)
        
        try:
            with open(path, 'w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(
                    file, 
                    fieldnames=['test_number', 'reaction_time', 'intensity', 
                               'color_component', 'user_input', 'correct']
                )
                writer.writeheader()
                writer.writerows(self.results)
            
            QMessageBox.information(self, "Sukces", 
                                  f"Wyniki zostały zapisane do pliku:\n{filename}")
        except Exception as e:
            QMessageBox.critical(self, "Błąd", 
                               f"Nie udało się zapisać wyników do pliku:\n{str(e)}")

    def display_results(self) -> None:
        self.figure.clear()
        self._create_reaction_time_plot()
        self._create_intensity_plot()
        self.figure.tight_layout()
        self.canvas.draw()

    def _create_reaction_time_plot(self) -> None:
        ax = self.figure.add_subplot(211)
        test_numbers = [r['test_number'] for r in self.results]
        reaction_times = [r['reaction_time'] for r in self.results]
        
        ax.plot(test_numbers, reaction_times, 'b-o', label='Czas reakcji')
        ax.set_xlabel('Numer testu')
        ax.set_ylabel('Czas reakcji (s)')
        ax.set_title('Czasy reakcji w kolejnych testach')
        ax.grid(True)

    def _create_intensity_plot(self) -> None:
        ax = self.figure.add_subplot(212)
        test_numbers = [r['test_number'] for r in self.results]
        intensities = [r['intensity'] for r in self.results]
        colors = self._get_component_colors()
        
        ax.bar(test_numbers, intensities, color=colors)
        ax.set_xlabel('Numer testu')
        ax.set_ylabel('Próg intensywności')
        ax.set_title('Progi intensywności w kolejnych testach')
        ax.grid(True)

    def _get_component_colors(self) -> List[str]:
        return ['red' if c == 'R' else 'green' if c == 'G' else 'blue' 
                for c in [r['color_component'] for r in self.results]]

    def adjust_image_intensity(self, image: npt.NDArray, intensity: float) -> npt.NDArray:
        """
        Dostosowuje intensywność wybranej składowej koloru w obrazie.
        
        Args:
            image: Obraz wejściowy w formacie RGB
            intensity: Wartość intensywności w zakresie 0-255
            
        Returns:
            Obraz z dostosowaną intensywnością wybranej składowej
        """
        adjusted = image.copy()
        
        if self.color_component == 'R':
            adjusted[:,:,0] = np.clip(adjusted[:,:,0] * (intensity / 255.0), 0, 255)
        elif self.color_component == 'G':
            adjusted[:,:,1] = np.clip(adjusted[:,:,1] * (intensity / 255.0), 0, 255)
        elif self.color_component == 'B':
            adjusted[:,:,2] = np.clip(adjusted[:,:,2] * (intensity / 255.0), 0, 255)
            
        return adjusted.astype(np.uint8)

    def display_test_image(self):
        if not self.test_images or self.current_test >= len(self.test_images):
            return
            
        current_image = self.test_images[self.current_test]
        
        adjusted_image = self.adjust_image_intensity(current_image, self.current_intensity)
        
        height, width = adjusted_image.shape[:2]
        bytes_per_line = 3 * width
        q_img = QImage(adjusted_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

    def handle_number_input(self):
        if not self.test_running:
            return
            
        number = self.number_input.text().strip()
        if not number:
            return
            
        self.timer.stop()
        reaction_time = time.time() - self.start_time
        
        current_image_name = f'it-{self.current_test + 1}.png'
        correct_number = IMAGE_NUMBERS.get(current_image_name, '')
        
        self.results.append({
            'test_number': self.current_test + 1,
            'reaction_time': reaction_time,
            'intensity': self.current_intensity,
            'color_component': self.color_component,
            'user_input': number,
            'correct': number == correct_number
        })
        
        self.number_input.clear()
        self.next_test()

    def update_color_intensity(self):
        self.current_intensity = min(255, self.current_intensity + self.INTENSITY_STEP)
        self.display_test_image()
        
        if time.time() - self.start_time > self.MAX_TEST_TIME:
            self.timer.stop()
            self.results.append({
                'test_number': self.current_test + 1,
                'reaction_time': self.MAX_TEST_TIME,
                'intensity': self.current_intensity,
                'color_component': self.color_component,
                'user_input': '',
                'correct': False
            })
            self.next_test()

    def load_next_test(self):
        if self.current_test >= self.max_tests:
            self.finish_test()
            return
            
        self.current_intensity = 0
        self.start_time = time.time()
        self.timer.start(40) 
        
        self.color_component = ['R', 'G', 'B'][self.current_test % 3]
        
        self.status_label.setText(
            f"Test {self.current_test + 1}/{self.max_tests}: "
        )
        self.display_test_image()

    def next_test(self):
        self.current_test += 1
        self.progress_bar.setValue(int(self.current_test / self.max_tests * 100))
        self.load_next_test()

    @property
    def is_test_complete(self) -> bool:
        """Sprawdza czy test został ukończony."""
        return self.current_test >= self.max_tests

    @property
    def test_progress(self) -> float:
        """Zwraca postęp testu w procentach."""
        return (self.current_test / self.max_tests) * 100

    def add_custom_image(self):
        """Handle adding a custom test image."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Wybierz obraz testowy",
            "",
            "Image files (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if not file_path:
            return
            
        number, ok = QInputDialog.getText(
            self,
            "Podaj prawidłową odpowiedź",
            "Jaka liczba jest ukryta na tym obrazie?",
            text=""
        )
        
        if not ok or not number.strip():
            return
            
        try:
            if not os.path.exists('test_images'):
                os.makedirs('test_images')
                
            next_num = self.max_tests + 1
            
            new_filename = f'it-{next_num}.png'
            new_path = os.path.join('test_images', new_filename)
            
            img = cv2.imread(file_path)
            if img is None:
                raise ValueError("Nie można wczytać obrazu")
                
            cv2.imwrite(new_path, img)
            
            self.update_image_numbers(new_filename, number)
            
            self.max_tests += 1
            self.test_images = []
            self.load_test_images()
            
            QMessageBox.information(
                self,
                "Sukces",
                f"Obraz został dodany jako {new_filename}\n"
                f"z przypisaną liczbą {number}"
            )
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Błąd",
                f"Nie udało się dodać obrazu:\n{str(e)}"
            )

    def update_image_numbers(self, image_name: str, number: str):
        """Update the image_numbers.py file with new image-number mapping."""
        try:
            try:
                with open('image_numbers.py', 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                if not content or content == 'IMAGE_NUMBERS = {\n} \n':
                    dict_content = {}
                else:
                    
                    import ast
                    dict_content = ast.literal_eval(content.split('=')[1].strip())
            except FileNotFoundError:
                dict_content = {}
            
            dict_content[image_name] = number
            
            with open('image_numbers.py', 'w', encoding='utf-8') as f:
                f.write('IMAGE_NUMBERS = {\n')
                for key, value in sorted(dict_content.items()):
                    f.write(f"    '{key}': '{value}',\n")
                f.write('} \n')
                
            import importlib
            import image_numbers
            importlib.reload(image_numbers)
            
        except Exception as e:
            raise Exception(f"Nie udało się zaktualizować pliku image_numbers.py: {str(e)}")

    def keyPressEvent(self, event):
        if not self.test_running:
            return
        
        if not self.number_mode.isChecked() and event.key() == Qt.Key_Space:
            self.timer.stop()
            reaction_time = time.time() - self.start_time
            
            self.results.append({
                'test_number': self.current_test + 1,
                'reaction_time': reaction_time,
                'intensity': self.current_intensity,
                'color_component': self.color_component,
                'user_input': 'SPACE',  
                'correct': True 
            })
            self.next_test()

    def handle_stop_resume(self):
        """Handle both stop and resume functionality"""
        if self.test_running:
            
            self.stop_test()
        else:
            
            self.test_running = True
            self.stop_button.setText("Przerwij test")
            self.number_input.setEnabled(True) if self.number_mode.isChecked() else self.setFocus()
            self.timer.start(40)

    def _reset_image_numbers(self) -> None:
        """Reset the image_numbers.py file to an empty dictionary."""
        try:
            with open('image_numbers.py', 'w', encoding='utf-8') as f:
                f.write('IMAGE_NUMBERS = {\n} \n')
                
            import importlib
            import image_numbers
            importlib.reload(image_numbers)
        except Exception as e:
            print(f"Warning: Could not reset image_numbers.py: {str(e)}")

def main():
    app = QApplication(sys.argv)
    test = ColorPerceptionTest()
    test.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main() 