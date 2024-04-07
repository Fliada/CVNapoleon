import streamlit as st
from stitch import SuperGlue
from io import BytesIO
from IPython.display import Image
from pathlib import Path
import numpy as np
import cv2
import os, shutil

super_glow = SuperGlue()

def delete_directory(folder: Path):
    for filename in os.listdir(str(folder.absolute())):
        file_path = os.path.join(str(folder.absolute()), filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    os.remove(str(folder.absolute()))

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
        
        if len(imgs) != 2: 
            st.warning("Ты должен выбрать только 2 фотографии для склейки!")
            return

        best_img = super_glow.connect_photos(photos=imgs)
        
        st.image(best_img[0], channels="BGR")

        path_img = super_glow.save_yolo(best_img[0]).absolute()

        st.image(str(path_img), channels="BGR")

        delete_directory(path_img.parent)

        with open(str(best_img[1].absolute()), "rb") as file:
            btn = st.download_button(
                    label="Скачать фотографию",
                    data=file,
                    file_name=best_img[1].name,
                    mime="image/jpeg"
                )
    

if __name__ == "__main__":
    main()