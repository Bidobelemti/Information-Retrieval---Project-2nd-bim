# Information Retrieval

Este proyecto implementa un sistema de recuperación de información aumentada. Permite el uso de texto o imagenes y, texto e imagenes, adicional de complementar los resultados con la implementación del modelo ___gemini-3-flash-preview___ de Google. Como corpus se usa el dataset presente en Kaggle [Amazon Consumer Reviews](https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products)

## Estructura del proyecto
```text
PROYECTO/
├── data/
│   ├──  1429.csv       <-- Archivo descargado obligatorio
|   ├──  Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv       <-- Archivo descargado obligatorio
|   ├──  Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv       <-- Archivo descargado obligatorio
|   └──img/   <-- Path donde se ubican todas las imagenes del corpus
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env
├── notebooks
│   ├── test.ipynb
│   ├── test.png
│   ├── test_2.png
│   └── test_3.png
└── src
    ├── UI
    │   └── interface.py
    ├── __init__.py
    ├── embeddings.py
    ├── loader_dataset.py
    ├── preprocessing.py
    ├── rag.py
    ├── search.py
    └── vector.py
```

## Integrantes 
* __Mauricio Morales__
* __Jossue Rivadeneira__ 

## Características
* Creación de un índice FAISS multimodal a partir de embeddings de imágenes y texto
* Uso de API de Gemini de Google para complementar los resultados del sistema.
* Uso de Gradio para interfaz gráfica que permite a un usario interactuar con el sistema

## Instalación

El proyecto esta realizado enteramente con Python.

1. Clonación de proyecto

```bash
git clone https://github.com/Bidobelemti/Information-Retrieval---Project-2nd-bim

cd Information-Retrieval---Project-2nd-bim
```
2. Descarga de dataset

Descargamos todos los archivos presentes en el dataset [Amazon Consumer Reviews](https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products)

3. Dependencias

Para la instalación de las dependencias necesarias para ejecutar el código es necesario realizar lo siguiente:

```bash
pip install -r requirements.txt
```
4. API de Google

Nos aseguramos de tener activa una api para Google Gemini, la podemos obtener en el siguiente URL https://aistudio.google.com/welcome

## Uso

El proyecto se ejecuta desde el archivo __main.py__ usando:

```bash
python main.py
```
Desde el navegador ingresamos a http://127.0.0.1:7860 URL donde se ejecuta la interfaz. Permite ingresar texto, subir una imagen o el uso de ambos métodos. El sistema nos devuelve 3 imagenes más relevantes, adicional de una respuesta generada por el LLM.



