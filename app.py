import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

def main():
    st.title("Streamlit App")
   
    @st.cache_resource
    def load_model():
        model = tf.keras.models.load_model('flower_classifier.hdf5')
        return model

    def import_and_predict(image_data, model):
        size = (128, 128)
        image = ImageOps.fit(image_data, size, Image.LANCZOS)
        img = np.asarray(image)
        img_reshape = img[np.newaxis, ...]
        prediction = model.predict(img_reshape)
        return prediction

    model = load_model()
    class_names = ["Daisy", "Dandelion", "Rose", "Sunflower", "Tulip"]

    st.write("# 🌸Flower Type Classifier Developed By Jerome Marbebe")
    st.write("# 🌼Daisy, 🌺Dandelion, 💐Tulips, 🌻Sunflower, 🥀Rose")

    file = st.file_uploader("Choose plant photo from computer", type=["jpg", "png"])

    if file is None:
        st.text("Please upload an image file")
    else:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        prediction = import_and_predict(image, model)
        class_index = np.argmax(prediction)
        class_name = class_names[class_index]
        string = "OUTPUT: " + class_name
        st.success(string)
 
if __name__ == "__main__":
    main()
