# FruitVision AI

FruitVision AI es un proyecto académico y técnico orientado a demostrar la aplicación de redes neuronales convolucionales para la clasificación del estado de frescura de frutas. El repositorio se apoya en el artefacto exportado real del modelo y en la evidencia observada directamente en el archivo Keras y su configuración.

## Objetivo académico

Construir una solución de visión artificial para reconocer cuatro clases:

- Manzana fresca
- Manzana deteriorada
- Banano fresco
- Banano deteriorado

La propuesta no busca reemplazar un proceso experimental completo, sino documentar de manera rigurosa la arquitectura real, el tipo de entrada/salida, el preprocesamiento requerido y la lógica de inferencia para un sistema de clasificación visual.

## 🌐 Acceso al proyecto

- **Landing Page:** [FRUTVISION IA]([https://dgonzro27.github.io/PROYECTO_ABP_ECO/](https://dgonzro27.github.io/PROYECTO_ABP_FRUITIA/))
- **Aplicación Streamlit:** [Probar FRUTIVISION IA](https://proyecto-abp-fruitia.streamlit.app/)

Explora la página de presentación del proyecto y accede a la aplicación para probar la clasificación de residuos mediante inteligencia artificial.

Explora la página de presentación del proyecto y accede a la aplicación para probar la clasificación de residuos mediante inteligencia artificial.

## Evidencia del modelo exportado

Se reutiliza el archivo exportado original en `models/keras_model.h5`, junto con `models/labels.txt`.

La verificación realizada sobre el artefacto exportado confirma que:

- El modelo recibe entradas de forma `224 x 224 x 3`.
- La salida final tiene dimensión 4, una por clase.
- La arquitectura exportada corresponde a una estructura basada en MobileNetV2 con `GlobalAveragePooling2D` y una capa densa final.
- El orden efectivo de la salida es:
  - 0: Manzana_fresca
  - 1: Manzana_deteriorada
  - 2: Banano_fresco
  - 3: Banano_deteriorado

Debido a que el archivo exportado de etiquetas tiene nombres truncados, la documentación usa una normalización semántica para preservar la intención del problema académico sin inventar información no observada.

## Proyecto y estructura del repositorio

```text
.
├── app.py
├── index.html
├── requirements.txt
├── README.md
├── models/
│   ├── keras_model.h5
│   └── labels.txt
├── dataset/
│   └── README.md
├── notebooks/
│   └── FruitVision_Redes_Neuronales.ipynb
├── src/
│   └── fruitvision/
│       ├── __init__.py
│       └── model_utils.py
├── .streamlit/
├── assets/
├── converted_keras_RN/
└── .venv/
```

## Instalación y ejecución

```bash
python -m venv .venv
. .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Para abrir la vista web del proyecto académico, también puede revisarse `index.html` como landing page y consultar la notebook Jupyter dentro de `notebooks/`.

## Descripción funcional

La aplicación en `app.py` permite:

- cargar una imagen desde disco o desde la cámara del usuario,
- preprocesarla con el esquema correcto para MobileNetV2,
- ejecutar inferencia con el modelo exportado,
- mostrar la clase predominante y su confianza,
- registrar un historial de predicciones por sesión.

## Metodología conceptual

El proyecto está pensado como una demostración del flujo completo de una solución de visión artificial:

1. comprensión del problema,
2. inspección del artefacto exportado,
3. preparación del input para la red,
4. inferencia y cálculo de probabilidades,
5. evaluación con métricas apropiadas,
6. documentación de limitaciones y alcance real.

## Evaluación y limitaciones honestas

This project explicitly avoids claiming metrics and accuracy values that are not supported by a real evaluation set. The repository does not include the original training dataset or the full training history, so any report of generalization or final accuracy must be produced with a separate test split and proper evaluation protocol.

Las limitaciones reales del proyecto son:

- No se dispone del dataset original ni del historial completo de entrenamiento.
- El archivo `labels.txt` exportado presenta nombres truncados al momento de la serialización.
- La evaluación final debe realizarse con un conjunto de prueba independiente.
- La arquitectura y el comportamiento del modelo se basan en la exportación disponible y no en un modelo reentrenado o reconstruido hipotéticamente.

## Separación de responsabilidades

- `app.py`: aplicación Streamlit de uso directo.
- `src/fruitvision/model_utils.py`: validación de etiquetas y preprocesamiento de imágenes.
- `models/`: artefactos del modelo exportado y sus etiquetas.
- `notebooks/FruitVision_Redes_Neuronales.ipynb`: documento técnico y didáctico del proyecto.
- `index.html`: landing page académica.
- `dataset/README.md`: detalles del conjunto de datos y sus limitaciones.

## Referencias conceptuales

- Google Teachable Machine
- TensorFlow / Keras
- MobileNetV2
- Redes neuronales convolucionales
- Aprendizaje profundo para visión artificial

## Estado final

El proyecto queda en una versión sólida, consistente con la exportación real, documentada de forma académica y lista para demostración local. La responsabilidad metodológica del equipo se mantiene en la evidencia disponible y en la comunicación honesta de los límites del artefacto exportado.
