# CVNapoleon
Кейс от NapoleonIT

Структура документа:<br />
.<br />
├── cv_models/<br />
│   ├── front.py&emsp;&emsp;&emsp;&emsp;&emsp;(фронт)<br />
│   ├── stitch.py&emsp;&emsp;&emsp;&emsp;(класс для преобразования фотографий)<br />
│   └── testing.ipynb&emsp;&emsp;&emsp;(notebook для тестирования класса)<br />
├── hack_stitch_dataset/&emsp;&emsp;(датасет для обучения)<br />
│   └── ...<br />
└── product-detection-in-shelf-yolov8  (дообученная модель)<br />

Чтобы запустить фронт нужно ввести команду:
```python
!streamlit run cv_model\front.py
```