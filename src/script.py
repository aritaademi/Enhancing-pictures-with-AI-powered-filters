    
# # Summarize challenges faced and solutions.
# # Disa nga veshtiresite qe hasa gjate projektit ishin: 
# # - Gjatë përpunimit të imazheve të mëdha, aplikimi i disa filtrave mund të shkaktonte ngadalësim të performancës 
# # - aplikimi i filterit cartoonize u desht te perdor algoritm te optimizuar si k-means për color quantization dhe bilateral filtering për të ruajtur balancimin mes cilësisë dhe performancës.

# # Highlight key decisions made during implementation.
# # - Përdorimi i strukturës së klasave për modularizimin e funksionaliteteve - Kjo ishte një zgjedhje kyçe për të organizuar projektin dhe për të bërë që çdo pjesë të ishte më e menaxhueshme dhe e testueshme
# # - Zgjedhja e bibliotekës OpenCV për përpunimin e imazheve - Zgjedhja e saj mundësoi përdorimin e funksioneve të avancuara si filtrat Gaussian, k-means për ndarje të ngjyrave dhe detektimin e kontureve, duke ofruar një gamë të gjerë opsionesh për manipulimin e imazheve.
# # - Integrimi i një ndërfaqe grafike për përdoruesit - Zgjedhja e përdorimit të PyQt për ndërtimin e ndërfaqes grafike ishte një vendim kyç. Kjo mundësoi krijimin e një mjedisi miqësor për përdoruesit me mundësinë për të përdorur butona dhe mundësi të thjeshta për të aplikuar filtrat dhe operacionet e ndryshme në imazhe.

import sys
import cv2  # Për manipulim imazhesh
import numpy as np  # Për përpunim numerik të dhënash
from PyQt5.QtWidgets import (QApplication, QMainWindow, QDockWidget, QLabel, QPushButton, QFileDialog,
                             QVBoxLayout, QWidget, QScrollArea, QSizePolicy, QGroupBox, QHBoxLayout, 
                             QProgressBar, QMessageBox)     # Komponentë të ndërfaqes
from PyQt5.QtGui import QPixmap, QImage    # Për të punuar me imazhe në PyQt5
from PyQt5.QtCore import Qt, QThread, pyqtSignal    # Për procese dhe sinjale në PyQt5
from filter_image import FilterImage    # Klasa për filtra të imazheve
import os   # Për menaxhim të skedarëve dhe dosjeve

# # Definohet rruga për dosjen 'images' brenda 'scripts'
# scripts_dir = os.path.dirname(os.path.abspath(__file__))
# images_dir = os.path.join(scripts_dir, "images")

# # Sigurohet ekzistenca e dosjes 'images'
# os.makedirs(images_dir, exist_ok=True)

# # Klasë për të krijuar një fije të veçantë për filtrim të imazhit
class FilterThread(QThread):
    update_progress = pyqtSignal(int)  # Sinjal për përditësimin e progresit (përcjell vlera të progresit në përqindje)
    filter_complete = pyqtSignal(object)  # Sinjal që dërgohet kur përfundon procesi i filtrimit (kalon objektin e imazhit të filtruar)

    # Konstruktor për inicializimin e klasës
    def __init__(self, filter_func, image): #__init__: Merr si parametra funksionin e filtrimit dhe imazhin për t'u përpunuar.
        super().__init__()    # Thërret konstruktorin e klasës bazë (QThread)
        self.filter_func = filter_func     # Ruhet funksioni i filtrimit
        self.image = image    # Ruhet imazhi që do të filtrohet

    def run(self):  # Funksioni që ekzekutohet kur fijeja fillon (metoda run e QThread)
        filtered_image = self.filter_func(self.image, self.update_progress)  # Ekzekuton funksionin e filtrimit duke kaluar imazhin dhe sinjalin e progresit
        self.filter_complete.emit(filtered_image)      # Lëshon sinjalin filter_complete pasi filtri ka përfunduar, duke dërguar imazhin e filtruar

# Klasë për editorin e imazheve
class ImageEditor(QMainWindow):
    def __init__(self):
        super().__init__()  # Inicializimi i klasës bazë QMainWindow
        self.image = None   # Variabël për të mbajtur imazhin aktual
        self.history = []  # Lista për historikun e imazheve (funksionaliteti Undo)
        self.filter_image = FilterImage()  # Instancë e klasës që përmban funksione për filtra, eshte pergjegjese per filtrat
        self.roi = None  # Rajoni e interesit (Region of Interest - ROI)
        self.initUI()  # Inicializon ndërfaqen grafike

    def initUI(self):
        self.setWindowTitle('Editor i Avancuar i Imazheve')    # Titulli i dritares
        self.setGeometry(50, 50, 1000, 700)      # Vendosja e dimensioneve të dritares

        self.label = QLabel(self)   # Element për shfaqjen e imazhit
        self.label.setAlignment(Qt.AlignCenter)     # Vendosje në qendër
        self.label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)  # Përmasa të adaptueshme

        self.scrollArea = QScrollArea(self)      # Zonë e lëvizshme për shfaqje, për lëvizje dhe përshtatje të imazhit
        self.scrollArea.setWidget(self.label)   # Vendos QLabel si widget për lëvizje
        self.scrollArea.setWidgetResizable(True)

        self.setCentralWidget(self.scrollArea)  # Vendos scrollArea si komponent kryesor të dritares

        self.sidebar = QDockWidget("Veglat", self)    # Pjesa anësore për veglat e punës
        self.sidebar.setAllowedAreas(Qt.LeftDockWidgetArea) # Vendoset në anën e majtë

        sidebar_widget = QWidget()      # Widget për mbajtjen e veglave
        sidebar_layout = QVBoxLayout()  # Layout vertikal për veglat

        # Add progress bar - Shiriti i progresit
        self.progressBar = QProgressBar(self)
        self.progressBar.setVisible(False)  # Nuk shfaqet fillimisht
        sidebar_layout.addWidget(self.progressBar)

        # File operations group
        file_group = QGroupBox("Operacionet me skedarë")   # Grupi i operacioneve për dosje
        file_layout = QHBoxLayout()     # Layout horizontal për butonat
        self.openButton = QPushButton('Hape foton', self)   # Butoni për hapje imazhi
        self.openButton.clicked.connect(self.openImage)     # Lidhet me funksionin për hapje
        self.saveButton = QPushButton('Ruaje foton', self)   # Butoni për ruajtje imazhi
        self.saveButton.clicked.connect(self.saveImage)      # Lidhet me funksionin për ruajtje
        file_layout.addWidget(self.openButton)
        file_layout.addWidget(self.saveButton)
        file_group.setLayout(file_layout)
        sidebar_layout.addWidget(file_group)

        # Butoni për kthim prapa (undo)
        self.backButton = QPushButton('Hiq filtrin', self)  # Krijojme butonin per heqjen e filtrit
        self.backButton.setEnabled(False)  # Fillimisht i çaktivizuar
        self.backButton.clicked.connect(self.undoLastFilter)  # Lidhet me funksionin për kthim prapa
        sidebar_layout.addWidget(self.backButton)

        # Grupi i filtrave bazikë
        basic_filters_group = QGroupBox("Filtrat e thjeshtë")    # Grupi për filtrat bazikë
        basic_filters_layout = QVBoxLayout()     # Layout vertikal për butonat e filtrave

        self.gaussianButton = QPushButton('Filtri Zbutja Gaussian / Turbullim', self)    # Butoni për Gaussian Blur
        self.gaussianButton.clicked.connect(self.apply_gaussian_filter)
        basic_filters_layout.addWidget(self.gaussianButton)

        self.highPassButton = QPushButton('Filtri Kalim i Lartë', self) # Butoni për filtrin High-Pass, Nënkupton një filtër që lejon frekuencat e larta (ndryshimet e mprehta në intensitet) të kalojnë, ndërsa bllokon frekuencat e ulëta
        self.highPassButton.clicked.connect(self.apply_high_pass_filter)
        basic_filters_layout.addWidget(self.highPassButton)

        self.lowPassButton = QPushButton('Filtri Kalim i ulët', self)   # Butoni për filtrin Low-Pass
        self.lowPassButton.clicked.connect(self.apply_low_pass_filter)
        basic_filters_layout.addWidget(self.lowPassButton)

        self.meanFilterButton = QPushButton('Filtri Mesatar', self)    # Butoni për Mean Filter
        self.meanFilterButton.clicked.connect(self.apply_mean_filter)
        basic_filters_layout.addWidget(self.meanFilterButton)

        basic_filters_group.setLayout(basic_filters_layout)
        sidebar_layout.addWidget(basic_filters_group)

        # Grupi i filtrave te avancuar
        advanced_filters_group = QGroupBox("Filtrat e Avancuar")
        advanced_filters_layout = QVBoxLayout()

        self.histogramButton = QPushButton('Barazimi i Histogramit', self)  # Butoni për ekuilibrim të histogramit, përdoret për të rritur kontrastin në një imazh, duke rishpërndarë intensitetet e pikselave në mënyrë të tillë që të shfrytëzohen më mirë të gjitha vlerat e mundshme nga 0 në 255 (për imazhet 8-bit)
        self.histogramButton.clicked.connect(self.apply_histogram_filter)
        advanced_filters_layout.addWidget(self.histogramButton)

        self.contourButton = QPushButton('Zbulo Konturat', self)   # Butoni për detektim të kontureve
        self.contourButton.clicked.connect(self.apply_contour_detection)
        advanced_filters_layout.addWidget(self.contourButton)

        self.noiseButton = QPushButton('Shto Zhurmë', self)   # Butoni për shtimin e zhurmës, metodë për të aplikuar modifikime në një imazh, në këtë rast për të shtuar një lloj zhurme specifike, si zhurma Gaussian
        self.noiseButton.clicked.connect(self.apply_add_noise)
        advanced_filters_layout.addWidget(self.noiseButton)

        self.minMaxButton = QPushButton('Zbutja me Min-Max', self)   # Butoni për zbutje Min-Max
        self.minMaxButton.clicked.connect(self.apply_min_max_smoothing)
        advanced_filters_layout.addWidget(self.minMaxButton)
        #Zbutja (Smoothing): Procesi i pastrimit të imazhit nga variacionet e papritura në intensitetin e pikselave, duke zbutur kalimet e papritura të ngjyrave ose intensitetit.
        #Min-Max: Ky proces përfshin gjetjen e vlerës minimale dhe maksimale të intensitetit të pikselave fqinjë në një rajon të caktuar (p.sh., një dritare 3x3), 
        # dhe pastaj përdorimin e këtyre vlerave për të krijuar një vlerë të re për pikselin që po shqyrtohet.

        self.medianFilterButton = QPushButton('Filtri Median', self)    # Butoni për Median Filter, një teknikë për përpunimin e imazheve që përdor medianën e intensitetit të pikselave fqinjë për të zëvendësuar vlerën e pikselit të caktuar. Ky proces ndihmon në largimin e zhurmave
        self.medianFilterButton.clicked.connect(self.apply_median_filter)
        advanced_filters_layout.addWidget(self.medianFilterButton)

        self.hybridMedianButton = QPushButton('Filtri Median Hibrid', self)     # Butoni për Hybrid Median Filter
        self.hybridMedianButton.clicked.connect(self.apply_hybrid_median_filter)
        advanced_filters_layout.addWidget(self.hybridMedianButton)
        #është një teknikë për përpunimin e imazheve që kombinon dy filtra median të ndryshëm për të përmirësuar cilësinë e imazhit 
        # dhe për të hequr zhurmat. Ky filtër përdor dy maska të ndryshme për të filtruar një imazh dhe pas filtrimit të dyja, merret mediana e rezultateve për të krijuar efektin përfundimtar.
        #Ky proces realizohet duke aplikuar një filtër median me një maskë dhe një filtër median tjetër me një maskë tjetër (me ndihmën e maskave të ndryshme, si p.sh., 3x3 ose 5x5).
        #  Pasi të aplikohet secili filtër, rezultati final është mediana e të dyja rezultateve të filtrave.

        self.cartoonizeButton = QPushButton('Filtri për Vizualizim', self)   # Butoni për stilizim e fotos si cartoon
        self.cartoonizeButton.clicked.connect(self.apply_cartoonize)
        advanced_filters_layout.addWidget(self.cartoonizeButton)

        advanced_filters_group.setLayout(advanced_filters_layout)
        sidebar_layout.addWidget(advanced_filters_group)

        sidebar_layout.addStretch(1)  # Hapësirë për të shtyrë veglat lart
        sidebar_widget.setLayout(sidebar_layout)    # Vendosim layout dhe widget-et për veglat e anësore
        self.sidebar.setWidget(sidebar_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.sidebar)     # Shton sidebar-in

        self.image = None   # Inicializimi i imazhit si None

    # Funksioni për hapjen e imazhit nga skedari
    #Përdor QFileDialog për të zgjedhur skedarin dhe më pas e lexon me OpenCV duke përdorur cv2.imread(). 
    def openImage(self):
        options = QFileDialog.Options() #përmban opsionet për dialogun e skedarëve. Këto opsione mund të konfigurohen për të përcaktuar mënyrën se si dialogu shfaqet 
        fileName, _ = QFileDialog.getOpenFileName(self, 'Open Image', '', 'Images (*.png *.xpm *.jpg *.jpeg *.bmp)', options=options) #hap një dritare dialogu për zgjedhjen e skedarëve
        if fileName:
            self.image = cv2.imread(fileName)   # Lexon imazhin duke përdorur OpenCV
            self.displayImage()  # Shfaq imazhin në ndërfaqe

    #Ruajtja e imazhit aktual me një emër të caktuar dhe zgjedhje të vendndodhjes.
    #Përdor QFileDialog për të zgjedhur destinacionin dhe e ruan imazhin me cv2.imwrite. 
    # Nëse nuk ka shtesë të skedarit, automatikisht shton .png.
    def saveImage(self):
        if self.image is not None:
            options = QFileDialog.Options()
            fileName, _ = QFileDialog.getSaveFileName( #opens a "Save File" dialog
                self, 'Save Image', os.path.join('images', 'filtered_image.png'), #The default path and filename.
                'Images (*.png *.xpm *.jpg *.jpeg *.bmp)', options=options #The filter, restricting the file types to common image formats
            )
            if fileName:
                if not (fileName.endswith('.png') or fileName.endswith('.jpg') or
                        fileName.endswith('.jpeg') or fileName.endswith('.bmp')):
                    fileName += '.png'
                try:
                    cv2.imwrite(fileName, self.image) #OpenCV writes the image data in self.image to the specified file.
                    print(f"Image saved successfully at {fileName}")
                except cv2.error as e:
                    print(f"Error: {e}")
                    print("Unable to save the image. Please ensure the filename has a correct extension and try again.")

    #Ruan  një kopje të imazhit aktual para se të aplikohen ndryshime, perdoret per funksionin Undo
    def saveImageState(self):
        """Saves the current image state into history."""
        if self.image is not None:      #Shton imazhin aktual në një histori (self.history) dhe aktivizon butonin "Hiq filtrin".
            self.history.append(self.image.copy())  # Krijojme nje kopje te fotos aktuale ne history
            self.backButton.setEnabled(True)  # E bejme 'enable' butonun Back/Hiq filtrin


    # Funksioni për aplikimin e Gaussian Filter me pyetje për rajonin
    def apply_gaussian_filter(self):
        if self.image is not None:
            self.saveImageState()  # Ruajme gjendjen e fotos para se te aplikojme filtrin
            reply = QMessageBox.question(
                self, 'Aplikimi i Filtrit',
                'Dëshironi ta aplikoni filtrin në të gjithë foton?',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.image = self.filter_image.gaussian_filter(self.image)
            else:
                self.apply_filter_to_region(self.filter_image.gaussian_filter)
            self.displayImage()

    # Funksioni për aplikimin e High-Pass Filter me pyetje për rajonin
    def apply_high_pass_filter(self):
        if self.image is not None:
            self.saveImageState()  # Ruajme gjendjen e fotos para se te aplikojme filtrin
            reply = QMessageBox.question(
                self, 'Aplikimi i Filtrit',
                'Dëshironi ta aplikoni filtrin në të gjithë foton?',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.image = self.filter_image.high_pass_filter(self.image)
            else:
                self.apply_filter_to_region(self.filter_image.high_pass_filter)
            self.displayImage()

#     # Funksioni për aplikimin e Low-Pass Filter me pyetje për rajonin
    def apply_low_pass_filter(self):
        if self.image is not None:
            self.saveImageState()  # Ruajme gjendjen e fotos para se te aplikojme filtrin
            reply = QMessageBox.question(
                self, 'Aplikimi i Filtrit',
                'Dëshironi ta aplikoni filtrin në të gjithë foton?',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.image = self.filter_image.low_pass_filter(self.image)
            else:
                self.apply_filter_to_region(self.filter_image.low_pass_filter)
            self.displayImage()

#     # Funksioni për aplikimin e Mean Filter me pyetje për rajonin
    def apply_mean_filter(self):
        if self.image is not None:
            self.saveImageState()  # Ruajme gjendjen e fotos para se te aplikojme filtrin
            reply = QMessageBox.question(
                self, 'Aplikimi i Filtrit',
                'Dëshironi ta aplikoni filtrin në të gjithë foton?',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.image = self.filter_image.mean_filter(self.image)
            else:
                self.apply_filter_to_region(self.filter_image.mean_filter)
            self.displayImage()

#     # Funksioni për aplikimin e Histogram Equalization me pyetje për rajonin
    def apply_histogram_filter(self):
        if self.image is not None:
            self.saveImageState()  # Ruajme gjendjen e fotos para se te aplikojme filtrin
            reply = QMessageBox.question(
                self, 'Aplikimi i Filtrit',
                'Dëshironi ta aplikoni filtrin në të gjithë foton?',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.image = self.filter_image.histogramme_filter(self.image)
            else:
                self.apply_filter_to_region(self.filter_image.histogramme_filter)
            self.displayImage()

#     # Funksioni për aplikimin e Detect Contours me pyetje për rajonin
    def apply_contour_detection(self):
        if self.image is not None:
            self.saveImageState()  # Ruajme gjendjen e fotos para se te aplikojme filtrin
            reply = QMessageBox.question(
                self, 'Aplikimi i Filtrit',
                'Dëshironi ta aplikoni filtrin në të gjithë foton?',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.image = self.filter_image.detect_contours(self.image)
            else:
                self.apply_filter_to_region(self.filter_image.detect_contours)
            self.displayImage()

#     # Funksioni për aplikimin e Add Noise me pyetje për rajonin
    def apply_add_noise(self):
        if self.image is not None:
            self.saveImageState()  # Ruajme gjendjen e fotos para se te aplikojme filtrin
            reply = QMessageBox.question(
                self, 'Aplikimi i Filtrit',
                'Dëshironi ta aplikoni filtrin në të gjithë foton?',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.image = self.filter_image.add_gaussian_noise(self.image)
            else:
                self.apply_filter_to_region(self.filter_image.add_gaussian_noise)
            self.displayImage()

#     # Funksioni për aplikimin e Min-Max Smoothing me pyetje për rajonin
    def apply_min_max_smoothing(self):
        if self.image is not None:
            self.saveImageState()  # Ruajme gjendjen e fotos para se te aplikojme filtrin
            reply = QMessageBox.question(
                self, 'Aplikimi i Filtrit',
                'Dëshironi ta aplikoni filtrin në të gjithë foton?',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.progressBar.setVisible(True)
                self.progressBar.setValue(0)
                self.thread = FilterThread(self.filter_image.min_max_smoothing, self.image)
                self.thread.update_progress.connect(self.updateProgress)
                self.thread.filter_complete.connect(self.filterComplete)
                self.thread.start()
            else:
                self.apply_filter_to_region_with_thread(self.filter_image.min_max_smoothing)
           

#     # Funksioni për aplikimin e Median Filter me pyetje për rajonin
    def apply_median_filter(self):
        if self.image is not None:
            self.saveImageState()  # Ruajme gjendjen e fotos para se te aplikojme filtrin
            reply = QMessageBox.question(
                self, 'Aplikimi i Filtrit',
                'Dëshironi ta aplikoni filtrin në të gjithë foton?',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.image = self.filter_image.median_filter(self.image)
                self.displayImage()
            else:
                self.apply_filter_to_region(self.filter_image.median_filter)
                self.displayImage()

#     # Funksioni për aplikimin e Hybrid Median Filter me pyetje për rajonin
    def apply_hybrid_median_filter(self):
        if self.image is not None:
            self.saveImageState()  # Ruajme gjendjen e fotos para se te aplikojme filtrin
            reply = QMessageBox.question(
                self, 'Aplikimi i Filtrit',
                'Dëshironi ta aplikoni filtrin në të gjithë foton?',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.progressBar.setVisible(True)
                self.progressBar.setValue(0)
                self.thread = FilterThread(self.filter_image.hybrid_median_filter, self.image)
                self.thread.update_progress.connect(self.updateProgress)
                self.thread.filter_complete.connect(self.filterComplete)
                self.thread.start()
            else:
                self.apply_filter_to_region_with_thread(self.filter_image.hybrid_median_filter)
            

#     # Funksioni për aplikimin e Cartoonize me pyetje për rajonin
    def apply_cartoonize(self):
        if self.image is not None:
            self.saveImageState()  # Ruajme gjendjen e fotos para se te aplikojme filtrin
            reply = QMessageBox.question(
                self, 'Aplikimi i Filtrit',
                'Dëshironi ta aplikoni filtrin në të gjithë foton?',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.image = self.filter_image.cartoonize_image(self.image)
            else:
                self.apply_filter_to_region(self.filter_image.cartoonize_image)
            self.displayImage()

  
    #Funksioni per te apliku rajonin e interesit ku duam te vendosim filtrin
    #Ky funksion  përdoret për të aplikuar një filtër në një rajon të veçantë të imazhit, 
    # që do të thotë një pjesë specifike e imazhit që zgjidhet nga përdoruesi. 
    def apply_filter_to_region(self, filter_function):
        roi = self.selectRegionForFilter()      #Përdor cv2.selectROI për të zgjedhur rajonin,  Nëse ROI është zgjedhur (nëse roi është e vlefshme), funksioni vazhdon.
        if roi is not None:     #Pasi të jetë zgjedhur ROI,
            x, y, w, h = roi    #funksioni merr koordinatat e tij , ku x dhe y janë pozitat e sipërme të majtë të drejtkëndëshit dhe w dhe h janë gjerësia dhe lartësia.
            region = self.image[y:y+h, x:x+w].copy()  #  pjesa përkatëse e imazhit (region) nxirret nga imazhi origjinal duke përdorur slicing (të dhëna të indeksuara të matrikseve).
            filtered_region = filter_function(region)  # Filtri aplikohet në këtë pjesë të imazhit duke përdorur funksionin filter_function(region), ku filter_function është një funksion që pranon si argument këtë region dhe kthen një version të përpunuar të tij
            self.image[y:y+h, x:x+w] = filtered_region  # Zëvendësoni pjesën origjinale
        if not hasattr(self, 'roi') or self.roi is None:
            print("No ROI selected. Please select a region first.")
            return

    #kontrollon për të siguruar që procesi të ndodhë në mënyrë të duhur, si verifikimi që ROI është zgjedhur dhe që dimensionet e filtrimit përputhen.
        # Nëse ndodhin gabime, ato kapen dhe shfaqen në mënyrë që përdoruesi të jetë i informuar për ndonjë problem.
        x, y, w, h = self.roi
        if w <= 0 or h <= 0:
            print("Invalid ROI selected!")
            return

        #menaxhon aplikimin e një filtri në një rajon të imazhit të zgjedhur, duke u siguruar që të gjitha hapat të ekzekutohen pa gabime
        try:
            # Nxjer rajonin e interesit,  Përdor slicing për të nxjerrë këtë rajon të veçantë të imazhit: self.image[y:y+h, x:x+w]. 
            # Kjo do të thotë se do të merren të gjitha rreshtat dhe kolonat nga pozita (x, y) deri në (x + w, y + h)
            region = self.image[y:y+h, x:x+w].copy()  # Sigurohemi qe kopjojme regjionin
            print(f"Original ROI shape: {region.shape}")

            # Aplikojme filrin ne rajonin e interesit ROI
            filtered_region = filter_function(region)   #Filtri që është kaluar si argument në funksion (filter_function) aplikohet mbi rajonin e zgjedhur.
            print(f"Filtered region shape: {filtered_region.shape}")

            # Kontrollon a eshte imazhi ne grayscale
            if len(filtered_region.shape) == 2:  # Grayscale
                filtered_region = cv2.cvtColor(filtered_region, cv2.COLOR_GRAY2BGR) 
                #cv2.cvtColor(filtered_region, cv2.COLOR_GRAY2BGR) përdoret për të konvertuar imazhin e filtruar në një imazh me 3 kanale (ngjyra RGB).
                #  Kjo është e nevojshme sepse imazhi origjinal mund të jetë me ngjyra (3 kanale), 
                # dhe për të ruajtur koherencën, duhet ta kthejmë imazhin e filtruar nga shkallë gri në formatin me 3 kanale.
            
            # Sigurohemi qe dimensionet pershtaten, nese jo do te shfaqet nje mesazh qe ka gabim
            if filtered_region.shape != region.shape:
                print("Filtered region dimensions do not match the ROI!")
                return

            # Updateon foton origjinale me rajonin e selektuar per filtrim
            self.image[y:y+h, x:x+w] = filtered_region  #Përdor slicing për të ndërruar vetëm atë pjesë të imazhit që është përpunuar dhe jo pjesën tjetër të imazhit. Kjo është mënyra se si filtri aplikohet në një rajon të veçantë pa prekur imazhin tjetër.
            print("Filter applied successfully to the selected region.")

        except Exception as e:
            print(f"Error applying filter: {e}")


#     # Funksioni për të aplikuar filtrin në një rajon të zgjedhur me përdorimin e fijeve
#është i lidhur me kodin paraprak. Dallimi është se ky përdor fije (threads) për të aplikuar filtra në një rajon të zgjedhur të imazhit, 
# që e bën më efikas për operacione të rëndë për mundësinë e ndërprerjes së funksionimit të aplikacionit dhe për të përmirësuar performancën.
    def apply_filter_to_region_with_thread(self, filter_func):
        roi = self.selectRegionForFilter()  #selectRegionForFilter() është funksioni që mundëson zgjedhjen e këtij rajoni.
        if roi is not None:
            x, y, w, h = roi    #përfaqësojnë koordinatat dhe përmasat e rajonit të zgjedhur, dhe pjesa e imazhit që përputhet me këto përmasa nxirret dhe ruhet si region.
            region = self.image[y:y+h, x:x+w].copy()  # Prerja e pjesës
            self.progressBar.setVisible(True)   #Ky hap është për të bërë të dukshëm dhe të mund të ndiqet progresi i filtrimit, duke e vendosur progresin fillestar në 0%. Kjo është e nevojshme për të informuar përdoruesin se operacioni është në zhvillim e sipër.
            self.progressBar.setValue(0)
            self.thread = FilterThread(filter_func, region) #Ky funksion përdor një fije (FilterThread) për të aplikuar filtrin në rajonin e zgjedhur.
            self.thread.update_progress.connect(self.updateProgress)    #Fijet përditësojnë progresin përmes signaleve, siç është update_progress, që është i lidhur me funksionin updateProgress për të rifreskuar progresin e përdoruesit.
            self.thread.filter_complete.connect(lambda filtered_region: self.updateRegion(x, y, w, h, filtered_region))
            self.thread.start()
        #FilterThread është një klasë që përmban logjikën për të aplikuar filtrin në mënyrë të pavarur dhe për të mundësuar përpunimin paralel.

#     # Funksioni për të përditësuar rajonin e filtruar
    #Funksioni kryen disa detyra për të përditësuar imazhin dhe për të bërë që ndryshimet të shfaqen për përdoruesin.
    #Argumentet e funksionit: x, y janë koordinatat fillestare të rajonit ku do të aplikohet filtri, w, h janë përmasat e rajonit të imazhit, filtered_region është variabli që mban vlerën e rezultatit të filtrit pas aplikimit.
    def updateRegion(self, x, y, w, h, filtered_region):  
        self.image[y:y+h, x:x+w] = filtered_region  #zëvendëson pjesën e imazhit në self.image me filtered_region, duke aplikuar filtrin në rajonin e përcaktuar nga x, y, w, h.
        self.displayImage() #Ky funksion siguron që të gjitha ndryshimet të jenë të dukshme për përdoruesin.
        self.progressBar.setVisible(False)

#     # Funksioni për të zgjedhur një rajon të fotos, ku me pas do te aplikohet filtri
    def selectRegionForFilter(self):
        if self.image is not None:
            # Konverto imazhin në format RGB për të përdorur cv2.selectROI
            image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            # Përdor cv2.selectROI për të zgjedhur rajonin
            #Parametri i parë është një dritare që shfaqet për të kërkuar përdoruesin të zgjedhë një pjesë të imazhit, 
            #Parametri False tregon që përdoruesi nuk mund të zgjidhë shumë rajone dhe nuk do të ketë mundësi të shfaqë ndihmë vizuale si shënues të lëvizshëm, Parametri i dytë False bën që dritarja të mos mbyllet automatikisht pas zgjedhjes.
            roi = cv2.selectROI("Zgjidh ku do ta aplikosh filtrin", image_rgb, False, False) 
            cv2.destroyWindow("Zgjidh ku do ta aplikosh filtrin")    #Pas përfundimit të zgjedhjes së rajonit, kjo linjë e kodit mbyll dritaren që ka shfaqur opsionin për zgjedhjen e ROI.
            if roi != (0, 0, 0, 0): #Funksioni kontrollon nëse ROI i zgjedhur nuk është (0, 0, 0, 0). Kjo është një vlerë që tregon se përdoruesi nuk ka zgjedhur asnjë rajon dhe ka mbyllur dritaren pa bërë një selektim.
                return roi  # (x, y, w, h)
            else:
                QMessageBox.information(self, 'Njoftim', 'Asnjë rajon i zgjedhur.')
                return None

#     # Ky funksion është përgjegjës për përditësimin e vlerës së shiritit të progresit gjatë procesit të aplikimit të një filtri.
    def updateProgress(self, value):        #value: Ky parametër tregon vlerën që do të vendoset në shiriti i progresit. Vlera zakonisht është një numër ndërmjet 0 dhe 100, ku 0 do të thotë që procesi nuk ka filluar dhe 100 tregon që procesi ka përfunduar.
        self.progressBar.setValue(value)
        #Ky rresht i kodit merr vlerën e përditësuar të progresit dhe e vendos në shiritin e progresit (self.progressBar). 
        # Funksioni setValue është një metodë që mundëson përditësimin e vlerës së shiritit të progresit në GUI (ndërfaqja grafike e përdoruesit).

    # Ky funksion është përgjegjës për përfundimin e procesit të filtrimit dhe përditësimin e imazhit përfundimtar në ndërfaqe. 
    # Ai përditëson gjithashtu shiritin e progresit dhe e fsheh atë pasi të ketë përfunduar filtrimi
    def filterComplete(self, result):   #result: Ky parametër është rezultati i filtrimit që është aplikuar në imazh. Pas përfundimit të filtrimit, result përmban imazhin e filtruar, i cili do të përdoret për të përditësuar pamjen e ndërfaqes.
        self.image = result #përditëson imazhin origjinal me imazhin e filtruar, duke i atribuar vlerën e filtruar të parametrin result të variablit self.image.
        self.displayImage()
        self.progressBar.setVisible(False)  #fshin shiritin e progresit nga ndërfaqja pasi filtrimi ka përfunduar dhe përpara se të shfaqet imazhi përfundimtar

    #Rikthen imazhin në gjendjen para filtrit të fundit, përdor një mekanizëm të historisë për të mbajtur gjurmët e imazheve të kaluara dhe për të mundësuar rikthimin e tyre
    #Konverton imazhin nga formati OpenCV (BGR ose grayscale) në formatin që kupton PyQt (QImage dhe QPixmap).
    def undoLastFilter(self):
        """Undo the last filter applied by restoring the last image state."""
        if self.history:    # çdo herë që një filtrim aplikohet, imazhi i mëparshëm ruhet në këtë listë për mundësinë e rikthimit.
            self.image = self.history.pop()  # Ky funksion heq dhe kthen elementin e fundit nga lista history. Kjo do të thotë që imazhi i fundit i ruajtur (që është imazhi i mëparshëm para filtrimit) do të rikthehet dhe do të përdoret si imazhi aktual.
            self.displayImage()
            if not self.history:  # Nese nuk ka asnje foto e ben 'enable' buton Back
                self.backButton.setEnabled(False)
        else:
            QMessageBox.information(self, 'Undo', 'No filter to undo.')

    #Rifreskon imazhin kur madhësia e dritares ndryshon.
    def displayImage(self):
        if self.image is not None:
            if len(self.image.shape) == 3:  # kontrollon  nëse imazhi është në formatin ngjyrë (BGR).
                h, w, _ = self.image.shape   # merr lartësinë (h) dhe gjerësinë (w) e imazhit. _ përdoret për të injoruar dimensionin e tretë që përfaqëson kanalet e ngjyrave (BGR).
                qformat = QImage.Format_RGB888  # Ky format është përshtatur për të dhënë imazhin në formatin RGB (për përdorim në Qt). 
            elif len(self.image.shape) == 2:  # Ky kontroll merret për të përpunuar imazhin nëse ai është shkallë gri (grayscale), ku ka vetëm dy dimensione (lartësinë dhe gjerësinë).
                h, w = self.image.shape     #Ky rresht merr vetëm lartësinë dhe gjerësinë e imazhit për shkak se imazhi është një matricë 2D (nuk ka kanale ngjyrash).
                qformat = QImage.Format_Grayscale8  #Ky është formati i duhur për imazhet në shkallë gri në Qt.
            else:
                return

            # krijon një QImage nga të dhënat e imazhit të ruajtur në self.image. 
            # Ky është një objekt që mund të përdoret në PyQt për të shfaqur imazhin.
            #self.image.data: Ky është një pointer në të dhënat e imazhit (një array numpy), self.image.strides[0]: Kjo është largësia në bytes që i referohet një rreshti të imazhit,
            #që përdoret për të siguruar përpunimin e duhur të të dhënave të imazhit, qformat: Formati i imazhit që do të përdoret (RGB për ngjyrën dhe shkallë gri për grayscale).
            img = QImage(self.image.data, w, h, self.image.strides[0], qformat)
            if len(self.image.shape) == 3:  #Ky hap është i nevojshëm sepse OpenCV përdor formatin BGR (Blue, Green, Red), ndërsa PyQt përdor RGB
                img = img.rgbSwapped()  # Konverton BGR ne RGB
            pixmap = QPixmap.fromImage(img)  # Krijon pixmap nga QImage
            #QPixmap është objekti që përdoret për të shfaqur një imazh në dritaren e PyQt.

            #Përshtat imazhin me madhësinë e re të etiketës (self.label).
            self.label.setPixmap(pixmap.scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)) 

    # Funksion per te update-u imazhin kur madhesia e dritares ndryshon
    def resizeEvent(self, event):
        if self.image is not None:
            self.displayImage() #i cili është përgjegjës për shfaqjen e imazhit dhe përshtatjen e tij me madhësinë e re të dritares.


# # Ekzekutimi kryesor i programit
if __name__ == '__main__':
    app = QApplication(sys.argv)    # Krijo aplikacionin PyQt5
    editor = ImageEditor()      # Inicializo editorin
    editor.show()            # Shfaq dritaren kryesore
    sys.exit(app.exec_())   # Mbyll aplikacionin kur përfundon