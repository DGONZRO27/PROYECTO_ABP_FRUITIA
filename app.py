from __future__ import annotations

import io
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.fruitvision.model_utils import MODEL_PATH, LABELS_PATH, load_label_names, preprocess_image_array

try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None

ROOT = Path(__file__).resolve().parent


@st.cache_resource
def load_model():
    """Load the exported Keras model once and cache it for the session."""
    if tf is None:
        raise RuntimeError("TensorFlow no está disponible en este entorno.")
    return tf.keras.models.load_model(MODEL_PATH)


st.set_page_config(page_title="FruitVision AI", page_icon="🍏", layout="wide")

st.title("FruitVision AI")
st.caption("Clasificación de frescura de frutas mediante redes neuronales convolucionales")

if "history" not in st.session_state:
    st.session_state.history = []

left, right = st.columns([1.2, 1.2])

with left:
    uploaded = st.file_uploader(
        "Sube una imagen de fruta",
        type=["jpg", "jpeg", "png", "webp"],
    )
    camera = st.camera_input("O captura con la cámara")

with right:
    st.markdown("### Información del modelo")
    st.write(f"Archivo del modelo: `{MODEL_PATH.name}`")
    st.write(f"Archivo de etiquetas: `{LABELS_PATH.name}`")
    st.write("Arquitectura real verificada: MobileNetV2 + GlobalAveragePooling2D + capa densa final de 4 clases.")
    st.write("El modelo se reutiliza sin reentrenar ni alterar sus pesos.")

image = None
if uploaded is not None:
    image = Image.open(uploaded)
elif camera is not None:
    image = Image.open(io.BytesIO(camera.getvalue()))

if image is not None:
    # `use_container_width` is not supported by older Streamlit versions; `width` is the
    # compatible cross-version option for displaying the uploaded image.
    st.image(image, caption="Imagen cargada", width=700)

    try:
        arr = np.array(image.convert("RGB"))
        processed = preprocess_image_array(arr, mode="mobilenetv2")
        processed = np.expand_dims(processed, axis=0)

        model = load_model()
        probabilities = model.predict(processed, verbose=0)[0]
        labels = load_label_names()

        pred_index = int(np.argmax(probabilities))
        pred_label = labels[pred_index]
        confidence = float(probabilities[pred_index])

        st.markdown("### Predicción")
        st.write(f"Clase: **{pred_label}**")
        st.write(f"Confianza: **{confidence:.2%}**")

        prob_pairs = sorted(
            zip(labels, probabilities.tolist()),
            key=lambda item: item[1],
            reverse=True,
        )
        chart_data = {label: float(prob) for label, prob in prob_pairs}
        st.bar_chart(chart_data)

        st.markdown("### Recomendación")
        if pred_label.endswith("fresco"):
            st.success("La fruta parece estar en buen estado de frescura. Se observa una apariencia saludable y consistente con una fruta fresca.")
        else:
            st.warning("La fruta muestra señales de deterioro. El sistema identifica que el estado es no fresco o deteriorado.")

        st.session_state.history.append({
            "clase": pred_label,
            "confianza": confidence,
            "probabilidades": {label: float(prob) for label, prob in zip(labels, probabilities)},
        })

        if len(st.session_state.history) > 10:
            st.session_state.history = st.session_state.history[-10:]

        st.markdown("### Historial de la sesión")
        hist_df = pd.DataFrame(st.session_state.history)
        if not hist_df.empty:
            st.dataframe(hist_df, use_container_width=True)
        else:
            st.write("Todavía no hay predicciones en esta sesión.")

    except Exception as exc:  # pragma: no cover
        st.error(f"No se pudo procesar la imagen: {exc}")

st.markdown("---")
st.markdown("### Sobre el funcionamiento de la red neuronal")
st.markdown(
    """
    La red toma la imagen como entrada, la preprocesa, extrae rasgos visuales con convoluciones y
    finalmente produce una distribución de probabilidad sobre cuatro clases: Manzana fresca,
    Manzana deteriorada, Banano fresco y Banano deteriorado.
    """
)
