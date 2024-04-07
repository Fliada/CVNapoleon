from pathlib import Path
from stitching.stitching import Stitcher
import cv2 
from stitching.stitching.camera_wave_corrector import WaveCorrector
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image
from YOLO import YOLOv8
from stitching.stitching import AffineStitcher

#Настройки для Stitcher
settings = {'detector': 'sift', 
    'confidence_threshold': 1e-05, 
    'finder': 'dp_color', 'adjuster': 
    'ray', 'warper_type': 'plane', 
    'block_size': 3, 'nfeatures': 15000, 
    'wave_correct_kind': 'horiz', 
    'match_conf': 0.4, 
    'crop': False, 
    'try_use_gpu': True
    }

CUR_PATH = Path('')

RESULT_PATH = CUR_PATH / "cv_model\\best_result.jpeg"

class SuperGlow:
    def __init__(self):
        #Инициализируем stitcher
        self.stitcher = Stitcher(**settings)
        #Выгружаем модель Yolo при инициализации класса
        if not (CUR_PATH / 'product-detection-in-shelf-yolov8').exists:
            assert Exception("No such directory 'product-detection-in-shelf-yolov8'. Please download the model")

        self.yolo_model = YOLOv8(detector_model_path=str((CUR_PATH / 'product-detection-in-shelf-yolov8\\best.pt').absolute()))
        
    def connect_photos(self, photos : list[Path] | list[Image], result_path : Path = RESULT_PATH) -> (Image, Path): # type: ignore
        """
        Соединяет несколько картинок в одну панораму
        return: лучшая панорама
        """

        images = self.preprocessing(photos=photos)
        best_img = self.__second__(self.__first__(images=images))
        
        cv2.imwrite(str(result_path.absolute()), best_img)

        return (best_img, result_path)

    def __to_img_list__(self, photo_paths : list[Path]) -> list[Image]:
        images = []
        for path in photo_paths: images.append(cv2.imread(path))
        return images

    #TO DO
    def __first__(self, images: list[Image]) -> list[Image]:
        """
        Первый слой обработки сгенерированных картинок с помощью YOLO\n
        return: список картинок с наибольших количеством задетекченных товаров
        """
        RANGE_NUM = 10
        min_supplies = 10e11
        min_supplies_index = 0
        _best_images_ = [(None, 1e10) for _ in range(int(RANGE_NUM / 2))]
        best_images = []

        for i in range(RANGE_NUM):
            res = self.stitcher.stitch(images=images)
            result = self.yolo_model.predict(res)
            # Подсчитываем количество товаров
            num_products = sum([len(result[i]) for i in range(len(result))])
            # print("Количество обнаруженных товаров:", num_products)

            if num_products < min_supplies:
                curr_min = 10e11
                _best_images_[min_supplies_index] = (res, num_products)
                for image_index in range(len(_best_images_)):
                    if (
                        _best_images_[image_index][1] < min_supplies
                        and _best_images_[image_index][1] < curr_min
                    ):
                        min_supplies = _best_images_[image_index][1]
                        min_supplies_index = image_index
                        curr_min = _best_images_[image_index][1]

        for image_index in range(len(_best_images_)):
            if _best_images_[image_index][0] is None: 
                continue
            best_images.append(_best_images_[image_index][0])

        return best_images


    def __second__(self, images : list[Image]) -> Image:
        """
        Второй слой обработки сгенерированных картинок с помощью определения количества пустот\n
        return: лучшая склеенная картинка
        """
        result_best = None
        best_black_pixels = 1e11

        for img in images:
            black_pixels = self.__calc_black_pixels__(img)
            if black_pixels < best_black_pixels:
                best_black_pixels = black_pixels
                result_best = img

        return result_best


    #TO DO
    def preprocessing(self, photos : list[Path] | list[Image]) -> list[Image]:
        """
        Предобработка картинок. Меняет ориентацию и угол картинки, если это необходимо
        """
        images = None
        if type(photos) is list[Path]:
            images = self.__to_img_list__(photo_paths=photos)
        else:
            images = photos

        return images

    def __calc_black_pixels__(self, image) -> int:
        """
        Вычисляет количество черных пикселей
        """
        return sum([1 for i in range(image.shape[0]) for j in range(image.shape[1]) if all(image[i][j] == [0, 0, 0])])    

    def show(self, img: Image):
        """
        Выводит картинку на экран
        """
        import matplotlib.pyplot as plt
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()

    def save_yolo(self, img: Image) -> Path:
        result = self.yolo_model.predict(image=img, project="preds", name="run")
        return CUR_PATH / 'preds' / 'run' / 'image0.jpg'
        