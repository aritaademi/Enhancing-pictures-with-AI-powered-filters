
import cv2  # bibliotekë për vizionin kompjuterik që ofron mjete për Përpunimin e imazheve (blur, detektim konturesh, filtrime), Operacione matematikore me imazhe (konvolucion, zbutje), Manipulimin e ngjyrave dhe histogramit
import numpy as np  #bibliotekë për përpunimin e të dhënave numerike, që ofron: Mbështetje për krijimin dhe manipulimin e matricave dhe arrays, Operacione për krijimin e kernelëve të filtrimit dhe zhurmave
from scipy import ndimage   #ndimage (nga SciPy): Një modul për përpunimin e të dhënave multidimensionale, që përdoret këtu për: Filtrime të specializuara, si filtri median hibrid.
import os   #Nuk është përdorur direkt në kodin e dhënë, por zakonisht përdoret për operacione me skedarë dhe direktoriume.

# Klasë që përmban metoda për filtrim dhe përpunim të imazheve
class FilterImage:
    def __init__(self, image=None):
        self.image = image

    # Funksioni për aplikimin e filtrit Gaussian per te be foton me te trubullt duke reduktuar detajet e mprehta dhe zhurmat
    #Është i bazuar në një funksion matematikor që krijon një shpërndarje Gaussian për të përcaktuar se si piksela afër ndikon në pikselin qendror
    #Kernel-i kalon mbi çdo piksel të imazhit.Për çdo piksel, merret një mesatare e peshuar e vlerave të intensitetit të pikselave fqinj, bazuar në peshat e kernel-it
    #o	Pikseli i ri merr një vlerë të "mesatarizuar", duke zbutur ndryshimet e mprehta.
    def gaussian_filter(self, image):
        if image is not None:
            return cv2.GaussianBlur(image, (15, 15), 0)  # cv2.GaussianBlur Aplikon blur Gaussian me kernel 15x15
        return image

    #Përmirëson skajet ose rajonet me frekuencë të lartë duke theksuar ndryshimet e papritura në intensitetin e pikselit.
    def high_pass_filter(self, image): #thekson detajet e mprehta ne imazh
        if image is not None:
            kernel = np.array([[-1, -1, -1], #Vlera qendrore (+8) përfaqëson ndikimin kryesor të pikselit që po përpunojmë
                               [-1,  8, -1], #Vlerat negative (-1) për fqinjët përfaqësojnë piksela që do të zbriten nga pikseli qendror.
                               [-1, -1, -1]])  # Çdo piksel fqinj shumëzohet me vlerën e tij në kernel, dhe pastaj të gjitha këto vlera mblidhen
            return cv2.filter2D(image, -1, kernel)  # Aplikon filtrin me kernel-in e dhënë  | Rezultati është një masë e diferencës midis pikselit qendror dhe fqinjëve të tij.
        return image

    #Rrit kontrastin e një imazhi duke rishpërndarë intensitetin e pikselit në mënyrë që histogrami të jetë më i përhapur.
    #Histogrami i një imazhi ℎ(𝐼) tregon frekuencën e çdo niveli intensiteti 𝐼'.
    def histogramme_filter(self, image):
        if image is not None:
            if len(image.shape) == 3:  # Ngjyra e fotos
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # konvertohet në shkallë gri (grayscale) sepse barazimi i histogramit zakonisht aplikohet mbi një kanal të vetëm
                equalized = cv2.equalizeHist(gray_image)  # cv2.equalizeHist për të barazuar histogramin e imazhit në shkallë gri.
                return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)  # Convert back to 3-channel
            elif len(image.shape) == 2:  # Grayscale image
                return cv2.equalizeHist(image)
        return image
    
    # Funksioni për zbulimin e kontureve në imazh duke përdorur metodën e algoritmit të Canny dhe funksione të tjera të OpenCV.
    def detect_contours(self, image): #ndryshimet e mëdha në intensitetin e pikselëve në një imazh do theksohen me ngjyre te gjelbert
        if image is not None:
            # Konverton në grayscale vetëm nëse imazhi nuk është grayscale
            if len(image.shape) == 3:  # Kontrollon nëse imazhi është me ngjyra (3 kanale)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #sepse konturet janë më të lehta për t'u zbuluar duke analizuar një kanal të vetëm intensiteti
            elif len(image.shape) == 2:  # Imazhi është tashmë grayscale
                gray = image
            else:
                return image

            blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Aplikon Gaussian blur për të zbutur imazhin dhe për të reduktuar zhurmat
            edges = cv2.Canny(blurred, 50, 150)  # Zbulon konturet me algoritmin Canny
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Gjen konturet
            cv2.drawContours(image, contours, -1, (0, 255, 0), 2)  # Vizaton konturet mbi imazh
            return image
        return image
    #Canny Edge Detection: Zbulon skajet duke ndjekur hapat:
    # Aplikimi i derivatit të parë për të llogaritur gradientin, për të llogaritur ndryshimin e intensitetit të pikselëve në të dy drejtimet x dhe y
    #Gjetja e pikave me ndryshim të madh (kufijtë) bazuar në vlera pragje Tmin dhe Tmax
    #Pikat që ndodhen midis këtyre vlerave pranohen si skaje nëse janë të lidhura me një pikë që kalon Tmax
    #cv2.RETR_EXTERNAL: Merr vetëm konturet e jashtme (p.sh., konturet e jashtme të objekteve).
    #cv2.CHAIN_APPROX_SIMPLE: Redukton numrin e pikave të konturit duke hequr pikat e tepërta 

    # Funksioni për shtimin e zhurmës Gaussian në imazh
    def add_gaussian_noise(self, image, progress_callback=None):
        if image is not None:
            if len(image.shape) == 3:  # Imazh me ngjyra
                row, col, ch = image.shape
                sigma = np.sqrt(0.01)  #Devijimi standard përcakton intensitetin e zhurmës.
                gauss = np.random.normal(0, sigma, (row, col, ch))
                noisy = image + gauss.reshape(row, col, ch) * 255 #shumzohet me 255 (për të përputhur intervalin e intensitetit të pikselëve 0−255).
            elif len(image.shape) == 2:  # Imazh grayscale
                row, col = image.shape
                sigma = np.sqrt(0.01)
                gauss = np.random.normal(0, sigma, (row, col))
                noisy = image + gauss * 255 #Për imazhet grayscale: Zhurma ka vetëm 1 kanal.
            
            # Kufizon vlerat në intervalin 0-255, siguron që çdo piksel të mos kalojë kufijtë e intensitetit
            noisy = np.clip(noisy, 0, 255).astype(np.uint8)
            
            if progress_callback:
                progress_callback(row * col)  # Kthen progresin nëse jepet callback thirret me numrin total të pikselëve të përpunuar (row×col)
            
            return noisy
        return image
    #Zhurma Gaussian: Gjenerohet nga një shpërndarje normale: Ku μ=0 është mesatarja dhe σ është devijimi standard. 
    # Vlerat e shpërndarjes normalizohen dhe shkallëzohen për t'u përshtatur me intensitetin e pikselave (0 deri 255).

    # Funksioni për filtrimin mesatar për të zbutur një imazh dhe për të reduktuar zhurmat.
    #zbut imazhin dhe redukton zhurmën duke zëvendësuar vlerën e secilit piksel me mesataren e fqinjëve të tij
    #  bazuar në një kernel (ose dritare) të përmasave n×m
    def mean_filter(self, image, n=3, m=3):
        if image is not None:
            kernel = np.ones((n, m), np.float32) / (n * m)  # Krijon kernel mesatar K(i,j) = 1/n*m ku n dhe m jane permasat e kernelit
            return cv2.filter2D(image, -1, kernel)  # Aplikon filtrin me kernel-in e dhënë, Për çdo piksel në imazh, intensiteti përditësohet si mesatarja e vlerave të fqinjëve:
        return image
    #Ky proces zbut kalimet e papritura në intensitet, duke ulur detajet e larta dhe zhurmat.

    # Funksioni për filtrin low-pass i cili zbut imazhin duke larguar zhurmat e larta frekuencë
    def low_pass_filter(self, image):
        if image is not None:      #Filtri përdor një kernel që është një matricë e shpërndarë dhe e përdor atë për të aplikuar një operacion konvolucioni mbi imazh.
            H = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16  #Shuma totale e elementeve është 1 (normalizim), që ruan ndriçimin e përgjithshëm të imazhit.
            return cv2.filter2D(image, -1, H)  # Aplikon filtrin me kernel-in e dhënë
        return image
    #H është një kernel 3×3 që ka vlera të ndara për të krijuar efektin e zbutjes
    #Elementët në qendër kanë vlerë më të madhe (4/16) për të dhënë më shumë peshë pikselëve të afërt gjatë përllogaritjes së mesatares
    #Filtri low-pass zbut imazhin duke mesatarizuar intensitetin e pikselëve bazuar në fqinjësinë 3×3

    #filter për të zbutur imazhin duke përdorur vlerat minimale dhe maksimale të fqinjëve të çdo piksela.
    #Mesatarja e këtyre dy vlerave mesatarja= min+max / 2 përdoret për të përditësuar intensitetin e pikselit qendror
    def min_max_smoothing(self, image, progress_callback=None):
        if len(image.shape) == 3: #Nëse imazhi është me ngjyra  funksioni e përpunon secilin kanal veçmas.
            result = np.zeros_like(image, dtype=np.uint8)  # Explicit dtype
            for i in range(3):  # Process each channel independently
                result[:, :, i] = self._min_max_smoothing_channel(image[:, :, i], progress_callback) #përpunon kanalin i duke thirrur funksionin ndihmës _min_max_smoothing_channel
            return result
        else:
            return self._min_max_smoothing_channel(image, progress_callback)
    #Për çdo piksel, Përcaktohet një zonë fqinjësie 3×3rreth pikselit, merret minimi dhe maksimumi i intensitetit të pikselave fqinj
    # Përdoret mesatarja e këtyre vlerave për të zëvendësuar vlerën e pikselit aktual

    #Funksioni _min_max_smoothing_channel përpunon një kanal të vetëm (grayscale) të një imazhi duke zbatuar zbutjen me metodën min-max.
    def _min_max_smoothing_channel(self, channel, progress_callback=None):
        rows, cols = channel.shape  #Merr madhësinë e kanalit (rreshtat dhe kolonat).
        result = np.zeros_like(channel, dtype=np.uint8)  # Krijohet një matricë bosh me të njëjtën madhësi si kanali, për të ruajtur vlerat e përpunuara.

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                neighborhood = channel[i-1:i+2, j-1:j+2]  # Për çdo piksel (i,j), merret një bllok fqinjësie 3×3 rreth tij.
                min_val = np.min(neighborhood) #Vlera minimale nga fqinjësia
                max_val = np.max(neighborhood) #Vlera maksimale nga fqinjësia
                result[i, j] = np.clip((np.float32(min_val) + np.float32(max_val)) / 2, 0, 255).astype(np.uint8)
            #Përllogaritet mesatarja midis min dhe max: mesatarja=min+max / 2, np.clip: Siguron që vlera mesatare të jetë në intervalin [0,255]
            #Rezultati konvertohet në uint8 për ruajtjen si piksel imazhi

            if progress_callback:
                progress_callback.emit(int((i / rows) * 100))  # Nëse është dhënë një callback për progresin, ai përditësohet bazuar në përqindjen e rreshtave të përpunuar

        return result #Kthehet matrica result, e cila përmban kanalin e përpunuar me zbutjen min-max


    # Funksioni  përdoret për të larguar zhurmat e imazhit duke zëvendësuar çdo piksel me median e intensitetit të fqinjëve të tij.
    def median_filter(self, image, size=3): #size: Madhësia e dritares ne kete rast dritare 3x3
        if image is not None:
            return cv2.medianBlur(image, size)  # Aplikon medianBlur, Çdo piksel zëvendësohet me median e fqinjëve të tij në dritaren size×size
        return image
    # Për çdo piksel (x,y), filtri median e zëvendëson atë me median e intensitetit të fqinjëve të tij: I ′(x,y)=median(N(x,y))
    #Ku 𝑁(𝑥,𝑦) është fqinjësia e pikselit (x,y). Ky operacion ndihmon për të ruajtur detajet e imazhit dhe për të larguar zhurmat.
   
   #bën filtrimin e dyfishtë me dy maska të ndryshme për çdo kanal të ngjyrës. 
   # Pas filtrimit me këto dy maska, merret median e rezultateve për të krijuar efektin përfundimtar.
    def hybrid_median_filter(self, image, progress_callback=None):
        if len(image.shape) == 3:
            result = np.zeros_like(image)
            for i in range(3):  # Process each channel separately
                result[:, :, i] = self._hybrid_median_channel(image[:, :, i]) #Për çdo kanal  përdoret funksioni ndihmës _hybrid_median_channel për të zbatuar filtrimin.
                if progress_callback:  # Nëse ofrohet një progress_callback, raportohet progresi gjatë procesimit të kanaleve.
                    progress_callback.emit(int((i + 1) / 3 * 100)) 
            return result
        else:
            result = self._hybrid_median_channel(image)
            if progress_callback:
                progress_callback.emit(100)  # Emit 100% progress for grayscale
            return result

    #Filtri përdor dy maska të ndryshme për të filtruar një imazh, dhe pas filtrimit, merret mediana e të dyja rezultateve: I′(x,y)=median(m1(x,y),m2(x,y),I(x,y))
    #Ku 𝑚1 dhe m2 janë filtrat median të aplikuar me maska të ndryshme, dhe 𝐼(𝑥,𝑦) është intensiteti origjinal i pikselit.
    def _hybrid_median_channel(self, channel):
        m1 = ndimage.median_filter(channel, footprint=np.array([[0,1,0],[1,1,1],[0,1,0]]))  # Zbaton median në pikselat vertikalë dhe horizontalë.
        m2 = ndimage.median_filter(channel, footprint=np.array([[1,0,1],[0,1,0],[1,0,1]]))  # Zbaton median në pikselat diagonalë.
        return np.median([m1, m2, channel], axis=0) #Kthehet kanali i përpunuar me filtrin hibrid
    
    # Funksioni për cartoonizimin e imazhit, jep efekt me vizuel
    def cartoonize_image(self, image):
        if image is not None:
            # Hapi 1: Redukton paletën e ngjyrave (kuantizim)
            data = image.reshape((-1, 3)) #Imazhi transformohet nga një matricë 2D ose 3D ne nje array ne formen Nx3, ku N eshte numri total i pikselave dhe secili piksel perfaqsohet nga tre vlera RGB
            data = np.float32(data) # te dhenat konvertohen ne float qe ti permbushin kerkesat e funksionit kmeans
            # Përdor K-means për reduktimin e ngjyrave
            K = 8  # Numri i grupeve të ngjyrave
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, labels, centers = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            centers = np.uint8(centers)
            quantized_image = centers[labels.flatten()] #Për çdo piksel në imazh, ngjyra e tij zëvendësohet me ngjyrën përfaqësuese të grupit.
            quantized_image = quantized_image.reshape(image.shape) #Reshape: Imazhi i kuantizuar rifithet në formatin origjinal

            # Hapi 2: Zbulon konturet me parametra të përmirësuar
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  #Konvertimi në shkallë gri për të thjeshtuar analizën e intensitetit
            gray_blurred = cv2.medianBlur(gray, 7)  #Përdoret median blur për të larguar zhurmat duke ruajtur skajet kryesore
            edges = cv2.adaptiveThreshold(
                gray_blurred, 255, #255: Vlera maksimale e pikselit në imazh binar
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # Metoda adaptative që llogarit mesataren e fqinjëve duke përdorur një peshë Gaussian, e cila përmirëson detektimin e skajeve në situata të ndryshme të ndricimit
                cv2.THRESH_BINARY, 9, 9 # Tipi i pragut që bën imazhin të bëhet binar (pikselat janë ose 0 ose 255
                #Parametrat 9, 9: Madhësia e fqinjësisë dhe konstanta C përdoren për të përcaktuar pragun.
            )
            # Hapi 3: Zbut rajonet e ngjyrave
        #Përdoret filtrimi bilateral për të zbutur imazhin duke ruajtur skajet, Zbut intensitetet brenda rajoneve uniforme,
        # Ruajtje e skajeve duke ndaluar ndërthurjen e pikselave me intensitete shumë të ndryshme
        #d=9: Madhësia e fqinjësisë së pikselave, σColor=75 : Ndjeshmëria ndaj ndryshimeve të ngjyrave, σSpace=75  Ndjeshmëria ndaj distancës hapësinore
            smoothed = cv2.bilateralFilter(quantized_image, d=9, sigmaColor=75, sigmaSpace=75)
            # Hapi 4: Kombinon konturet dhe rajonet e zbutura për efektin final, 
            # Konturet dhe rajonet e zbutura kombinohen duke përdorur operacionin bitwise AND, që jep efektin e vizatimit.
            #Rajonet me ngjyra (pas filtrimit bilateral) mbulohen me një maskë që përmban konturet e zbuluara.
            cartoon = cv2.bitwise_and(smoothed, smoothed, mask=edges)
            return cartoon
        return image
    
    #criteria: Kjo specifikon kushtet për përfundimin e algoritmit (p.sh., numri maksimal i iteracioneve ose një tolerancë e caktuar).
    #labels dhe centers: Rezultoni përfshin etiketat që tregojnë se cila grupë i përket çdo piksel dhe qendrat e grupeve që përfaqësojnë ngjyrat e përzgjedhura.
    
