# CVNapoleon
Кейс от NapoleonIT

Структура документа:<br />
.
├── cv_models/
│   ├── front.py                (фронт)
│   ├── stitch.py               (класс для преобразования фотографий)
│   └── testing.ipynb           (notebook для тестирования класса)
├── hack_stitch_dataset/        (датасет для обучения)
│   └── ...
└── product-detection-in-shelf-yolov8  (дообученная модель)

Чтобы запустить фронт нужно ввести команду:
```python
!streamlit run cv_model\front.py
```