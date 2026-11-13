# Dataset y procedencia

Este proyecto reutiliza el modelo exportado ya entrenado en `models/keras_model.h5`.
El conjunto de imágenes original no se almacena en este repositorio, por lo que este directorio
actúa como registro de la estructura esperada y de la limitación documental.

## Estructura esperada

- `train/`: imágenes de entrenamiento.
- `test/`: imágenes reservadas para evaluación independiente.
- `validation/`: conjunto opcional de validación.

Se recomienda mantener una separación estricta por origen y evitar duplicados de la misma imagen
entre train y test.

## Limitaciones documentadas

- La exportación disponible en este repositorio no incluye el dataset original ni el historial de
  entrenamiento.
- La clase guardada por Teachable Machine aparece truncada en `labels.txt` por el export; por ello,
  se interpreta con la nomenclatura normalizada del proyecto académico.
- No se puede verificar el número exacto de imágenes ni la separación original sin acceder al
  dataset fuente.

## Recomendación

Cuando el conjunto original esté disponible, se debe usar una partición por imagen-base o por grupo
de procedencia para evitar fugas de información entre entrenamiento y prueba.
