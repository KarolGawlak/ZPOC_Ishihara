import sys
import csv
import time
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QPushButton, QLabel, QFileDialog, QProgressBar,
                           QMessageBox, QHBoxLayout)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage, qRgb
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import cv2
import os
from typing import List, Dict, Any, Optional
import numpy.typing as npt

class ColorPerceptionTest(QMainWindow):
    """
    Aplikacja do testowania percepcji barw.
    
    Wyświetla serię obrazów testowych i mierzy czas reakcji użytkownika
    na rozpoznanie ukrytych liczb przy różnych poziomach intensywności
    składowych kolorów RGB.
    """

    # Stałe konfiguracyjne
    WINDOW_TITLE = "Test Percepcji Barw"
    WINDOW_GEOMETRY = (100, 100, 800, 600)
    IMAGE_SIZE = (400, 400)
    TIMER_INTERVAL = 40  # ms
    INTENSITY_STEP = 1.2
    MAX_TEST_TIME = 15  # seconds
    COLOR_COMPONENTS = ['R', 'G', 'B']
    
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(self.WINDOW_TITLE)
        self.setGeometry(*self.WINDOW_GEOMETRY)
        
        # Initialize variables
        self.current_test: int = 0
        self.max_tests: int = 10
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
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create test image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label)
        
        # Create progress bar and label
        progress_layout = QHBoxLayout()
        progress_label = QLabel("Postęp testu:")
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(progress_label)
        progress_layout.addWidget(self.progress_bar)
        layout.addLayout(progress_layout)
        
        # Create buttons layout
        button_layout = QHBoxLayout()
        
        # Start button
        self.start_button = QPushButton("Start testu")
        self.start_button.clicked.connect(self.start_test)
        self.start_button.setStyleSheet("QPushButton { padding: 10px; font-size: 14px; }")
        button_layout.addWidget(self.start_button)
        
        # Stop button
        self.stop_button = QPushButton("Przerwij test")
        self.stop_button.clicked.connect(self.stop_test)
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("QPushButton { padding: 10px; font-size: 14px; }")
        button_layout.addWidget(self.stop_button)
        
        layout.addLayout(button_layout)
        
        # Instructions layout
        instructions_layout = QVBoxLayout()
        
        # Create status label
        self.status_label = QLabel("Naciśnij 'Start testu' aby rozpocząć")
        self.status_label.setStyleSheet("QLabel { font-size: 14px; margin: 10px; }")
        instructions_layout.addWidget(self.status_label)
        
        # Add keyboard shortcut info
        shortcuts_label = QLabel("Naciśnij SPACJĘ gdy rozpoznasz liczbę na obrazie")
        shortcuts_label.setAlignment(Qt.AlignCenter)
        shortcuts_label.setStyleSheet("QLabel { font-size: 14px; margin: 10px; }")
        instructions_layout.addWidget(shortcuts_label)
        
        layout.addLayout(instructions_layout)
        
        # Setup timer for color intensity changes
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_color_intensity)
        
        # Setup results graph
        self.figure = Figure(figsize=(6, 4))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

    def load_test_images(self) -> None:
        try:
            self._load_images_from_directory()
        except FileNotFoundError as e:
            self._handle_missing_images_error(str(e))
        except Exception as e:
            self._handle_general_loading_error(str(e))

    def _load_images_from_directory(self) -> None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        image_dir = os.path.join(current_dir, 'test_images')
        
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Katalog {image_dir} nie istnieje")
        
        for i in range(1, self.max_tests + 1):
            self._load_single_image(image_dir, i)

    def _handle_missing_images_error(self, error_msg: str) -> None:
        QMessageBox.critical(self, "Błąd", f"Nie znaleziono katalogu z obrazami: {error_msg}")

    def _handle_general_loading_error(self, error_msg: str) -> None:
        """Obsługuje ogólne błędy podczas ładowania obrazów."""
        QMessageBox.critical(
            self, 
            "Błąd", 
            f"Wystąpił błąd podczas ładowania obrazów: {error_msg}"
        )

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
                    print(f"Nie udało się załadować obrazu: it-{index}.png")
            except Exception as e:
                print(f"Błąd podczas ładowania obrazu it-{index}.png: {str(e)}")
                raise
        else:
            print(f"Nie znaleziono pliku: {image_path}")
            if not self.test_images:  # Tylko jeśli nie mamy żadnych obrazów
                raise FileNotFoundError(f"Nie znaleziono pliku: {image_path}")

    def start_test(self):
        if not self.test_images:
            QMessageBox.warning(self, "Ostrzeżenie", "Brak dostępnych obrazów testowych!")
            return
            
        self.test_running = True
        self.current_test = 0
        self.results = []
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.load_next_test()

    def stop_test(self):
        self.test_running = False
        self.timer.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Test zatrzymany")

    def finish_test(self):
        self.test_running = False
        self.timer.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Test zakończony - wyświetlam wyniki")
        
        # Zapisz i wyświetl wyniki
        self.save_results()
        self.display_results()
        
        # Pokaż podsumowanie
        avg_time = sum(r['reaction_time'] for r in self.results) / len(self.results)
        avg_intensity = sum(r['intensity'] for r in self.results) / len(self.results)
        
        summary = (f"Test zakończony!\n\n"
                  f"Liczba wykonanych testów: {len(self.results)}\n"
                  f"Średni czas reakcji: {avg_time:.2f} s\n"
                  f"Średni próg intensywności: {avg_intensity:.1f}")
        
        QMessageBox.information(self, "Podsumowanie testu", summary)

    def save_results(self):
        # Upewnij się, że folder results istnieje
        results_dir = 'results'
        if not os.path.exists(results_dir):
            try:
                os.makedirs(results_dir)
            except Exception as e:
                QMessageBox.critical(self, "Błąd", 
                                   f"Nie można utworzyć katalogu results: {str(e)}")
                return

        # Utwórz pełną ścieżkę do pliku
        filename = f"wyniki_testu_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        path = os.path.join(
            results_dir, 
            f"wyniki_testu_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        
        try:
            with open(path, 'w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=['test_number', 'reaction_time', 
                                                        'intensity', 'color_component'])
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
        # Stwórz kopię obrazu
        adjusted = image.copy()
        
        # Dostosuj intensywność wybranej składowej koloru
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
            
        # Pobierz aktualny obraz testowy
        current_image = self.test_images[self.current_test]
        
        # Dostosuj intensywność
        adjusted_image = self.adjust_image_intensity(current_image, self.current_intensity)
        
        # Konwertuj na QImage
        height, width = adjusted_image.shape[:2]
        bytes_per_line = 3 * width
        q_img = QImage(adjusted_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # Wyświetl obraz
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

    def load_next_test(self):
        if self.current_test >= self.max_tests:
            self.finish_test()
            return
            
        self.current_intensity = 0
        self.start_time = time.time()
        self.timer.start(40)  # Zmniejszamy z 50ms na 40ms (20% szybciej)
        
        # Zmień składową koloru dla każdego testu
        self.color_component = ['R', 'G', 'B'][self.current_test % 3]
        
        self.status_label.setText(
            f"Test {self.current_test + 1}/{self.max_tests}: "
            f"Naciśnij SPACJĘ gdy rozpoznasz liczbę na obrazie"
        )
        self.display_test_image()

    def update_color_intensity(self):
        # Zwiększamy krok intensywności z 1 na 1.2
        self.current_intensity = min(255, self.current_intensity + 1.2)
        self.display_test_image()
        
        # Jeśli minęło 5 sekund od rozpoczęcia testu, przechodzimy do następnego
        if time.time() - self.start_time > 15:
            self.timer.stop()
            self.results.append({
                'test_number': self.current_test + 1,
                'reaction_time': 5.0,  # Maksymalny czas
                'intensity': self.current_intensity,
                'color_component': self.color_component
            })
            self.next_test()

    def next_test(self):
        self.current_test += 1
        self.progress_bar.setValue(int(self.current_test / self.max_tests * 100))
        self.load_next_test()

    def keyPressEvent(self, event):
        if not self.test_running:
            return
            
        if event.key() == Qt.Key_Space:
            # Użytkownik widzi obrazek
            self.timer.stop()
            reaction_time = time.time() - self.start_time
            self.results.append({
                'test_number': self.current_test + 1,
                'reaction_time': reaction_time,
                'intensity': self.current_intensity,
                'color_component': self.color_component
            })
            self.next_test()

    @property
    def is_test_complete(self) -> bool:
        """Sprawdza czy test został ukończony."""
        return self.current_test >= self.max_tests

    @property
    def test_progress(self) -> float:
        """Zwraca postęp testu w procentach."""
        return (self.current_test / self.max_tests) * 100

def main():
    app = QApplication(sys.argv)
    test = ColorPerceptionTest()
    test.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main() 