from pathlib import Path
from stitching.stitching import Stitcher, Images
import cv2
from stitching.stitching.camera_wave_corrector import WaveCorrector
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image
from YOLO import YOLOv8
from stitching.stitching import AffineStitcher

# Настройки для Stitcher
settings = {
    "detector": "sift",
    "confidence_threshold": 1e-05,
    "finder": "dp_color",
    "adjuster": "ray",
    "warper_type": "plane",
    "block_size": 3,
    "nfeatures": 2500,
    "wave_correct_kind": "auto",
    "match_conf": 0.4,
    "crop": False,
    "try_use_gpu": True,
}

CUR_PATH = Path("")

RESULT_PATH = CUR_PATH / "cv_model\\best_result.jpeg"


class SuperGlue:
    def __init__(self):
        # Инициализируем stitcher
        self.stitcher = Stitcher(**settings)
        # Выгружаем модель Yolo при инициализации класса
        if not (CUR_PATH / "product-detection-in-shelf-yolov8").exists:
            assert Exception(
                "No such directory 'product-detection-in-shelf-yolov8'. Please download the model"
            )

        self.yolo_model = YOLOv8(
            detector_model_path=str(
                (CUR_PATH / "product-detection-in-shelf-yolov8\\best.pt").absolute()
            )
        )

    def connect_photos(
        self, photos: list[Path] | list[Image], result_path: Path = RESULT_PATH
    ) -> (Image, Path):  # type: ignore
        """
        Соединяет несколько картинок в одну панораму
        return: лучшая панорама
        """

        images = self.preprocessing(photos=photos)
        best_img = self.__second__(self.__first__(images=images))

        cv2.imwrite(str(result_path.absolute()), best_img)

        return (best_img, result_path)

    def __to_img_list__(self, photo_paths: list[Path]) -> list[Image]:
        images = []
        for path in photo_paths:
            images.append(cv2.imread(path))
        return images

    # TO DO
    def __first__(self, images: list[Image]) -> list[Image]:
        """
        Первый слой обработки сгенерированных картинок с помощью определения количества черных пикселей
        return: список изображений с наименьшим количеством черных пикселей
        """

        RANGE_NUM = 10

        images = Images.of(images)
        images = list(images.resize(Images.Resolution.MEDIUM))
        masks = list(
            Images.of(
                [
                    cv2.inRange(image, np.array([0, 0, 1]), np.array([255, 255, 255]))
                    for image in images
                ]
            ).resize(Images.Resolution.MEDIUM)
        )

        # Генерируем изображения с помощью Stitcher
        generated_images = []
        for _ in range(RANGE_NUM):
            try:
                generated_images.append(
                    self.stitcher.stitch(images=images, masks=masks)
                )
            except Exception as _:
                pass
        # Генерируем 5 изображений
        # Создаем список кортежей, содержащих сгенерированное изображение и количество черных пикселей
        black_pixels_list = [
            (
                img,
                self.__calc_black_pixels__(
                    cv2.resize(img, (img.shape[1] // 10, img.shape[0] // 10))
                ),
            )
            for img in generated_images
        ]
        # Сортируем список по количеству черных пикселей
        sorted_images = sorted(black_pixels_list, key=lambda x: x[1])
        # Возвращаем только изображения с наименьшим количеством черных пикселей
        return [
            img for img, _ in sorted_images[: int(RANGE_NUM / 2)]
        ]  # Возвращаем изображения с наименьшим количеством черных пикселей

    def __second__(self, images: list[Image]) -> Image:
        """
        Второй слой обработки сгенерированных картинок с помощью YOLO
        return: изображение с наибольшим количеством обнаруженных товаров
        """
        # Инициализируем переменные для хранения наилучшего результата
        best_num_products = 0
        best_image = None

        # Проходимся по каждому изображению из списка
        for img in images:
            # Получаем результаты предсказания моделью YOLO
            result = self.yolo_model.predict(img)
            # Подсчитываем количество обнаруженных товаров
            num_products = sum([len(result[i]) for i in range(len(result))])
            # Если это изображение дает лучший результат, обновляем переменные
            if num_products > best_num_products:
                best_num_products = num_products
                best_image = img

        return best_image

    # TO DO
    def preprocessing(self, photos: list[Path] | list[Image]) -> list[Image]:
        """
        Предобработка картинок. Меняет ориентацию и угол картинки, если это необходимо
        """
        images = None
        if type(photos) is list[Path]:
            images = self.__to_img_list__(photo_paths=photos)
        else:
            images = photos

        return [self.__flip_if_should__(img) for img in images]

    def __flip_if_should__(self, img: Image):
        height, width, channels = img.shape
        if height > width:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        return img

    def __calc_black_pixels__(self, image) -> int:
        """
        Вычисляет количество черных пикселей
        """
        return sum(
            [
                1
                for i in range(image.shape[0])
                for j in range(image.shape[1])
                if all(image[i][j] == [0, 0, 0])
            ]
        )

    def show(self, img: Image):
        """
        Выводит картинку на экран
        """
        import matplotlib.pyplot as plt

        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()

    def save_yolo(self, img: Image) -> Path:
        self.yolo_model.predict(image=img, project="preds", name="run")
        return CUR_PATH / "preds" / "run" / "image0.jpg"
