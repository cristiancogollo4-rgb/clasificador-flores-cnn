import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 🔥 Configuración de la página
st.set_page_config(page_title="Clasificador de Flores", page_icon="🌸")

st.title("🌸 Clasificador de Flores")
st.write("Sube una imagen y el modelo predecirá el tipo de flor.")

# 🔥 Cargar modelo
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("modelo_flores.keras")

model = load_model()

# 🔥 Clases (IMPORTANTE: mismo orden)
class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

# 🔥 Subir imagen
uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    
    # Mostrar imagen
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada", use_column_width=True)
    
    # 🔥 Preprocesamiento
    img = image.resize((180, 180))
    img_array = np.array(img)
    
    # Asegurar 3 canales (por si es escala de grises)
    if img_array.shape[-1] != 3:
        img_array = np.stack((img_array,)*3, axis=-1)
    
    img_array = np.expand_dims(img_array, axis=0)
    
    # 🔥 Predicción
    predictions = model.predict(img_array)
    probs = tf.nn.softmax(predictions[0]).numpy()
    
    # 🔥 Mostrar probabilidades
    st.subheader("📊 Probabilidades por clase:")
    
    prob_dict = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
    st.bar_chart(prob_dict)
    
    # Mostrar porcentaje
    for clase, prob in prob_dict.items():
        st.write(f"{clase}: {prob:.2%}")
    
    # 🔥 Clase más probable
    pred_class = class_names[np.argmax(probs)]
    
    st.success(f"🌼 Predicción final: {pred_class}")
    
    