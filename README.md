# CVNapoleon
Кейс от NapoleonIT

Структура документа:<br />
|- cv_models/<br />
    | - front.py                        (фронт)<br />
    | - stitch.py                       (класс для преобразования фотографий)<br />
    | - testing.ipynb                   (notebook для тестирование класса)<br />
|- hack_stitch_dataset/                 (датасет для обучения)<br />
    | - ...<br />
|- product-detection-in-shelf-yolov8    (дообученная модель)<br />

Чтобы запустить фронт нужно ввести команду:
```python
!streamlit run cv_model\front.py
```