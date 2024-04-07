import streamlit as st
from stitch import SuperGlow
from io import BytesIO
from IPython.display import Image
from pathlib import Path
import numpy as np
import cv2 
import os

super_glow = SuperGlow()

def main():

    st.title('Склеивание нескольких фотографий')

    uploaded_files = st.file_uploader("Выберите несколько фотографий для склейки:", 
                                      accept_multiple_files=True, 
                                      type="JPEG"
                                      )

    imgs = []
    
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        st.write("Название файла:", uploaded_file.name)
        st.image(bytes_data)
        np_arr = np.frombuffer(bytes_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        imgs.append(image)

    if st.button('Преобразовать'):
        
        if len(imgs) < 2: 
            st.warning("Ты должен выбрать 2 или более фотографий")
            return

        best_img = super_glow.connect_photos(photos=imgs)
        
        st.image(best_img[0], channels="BGR")

        path_img = super_glow.save_yolo(best_img[0]).absolute()

        st.image(str(path_img), channels="BGR")

        os.remove(str(path_img / '../'))

        with open(str(best_img[1].absolute()), "rb") as file:
            btn = st.download_button(
                    label="Скачать фотографию",
                    data=file,
                    file_name=best_img[1].name,
                    mime="image/jpeg"
                )
    

if __name__ == "__main__":
    main()