from sentence_transformers import SentenceTransformer
from PIL import Image
import numpy as np
import os
import torch

def generate_text_embeddings (texts : list[str], model_name = "clip-ViT-B-32", device = None) -> np.ndarray:
    '''
    ## Input:
    - texts 

        Lista de strings que seran convertidos en CUDA
    - model_name

        Por defecto usamos clip-ViT-B-32, puede cambiar
    -device

        No es necesario colocar nada en el argumento de device, a menos que sea seguro que se ocupe CUDA, caso contrario el código lo activa
    ## Output:
    - np.ndarray

        Todo el conjunto de vectores que conpone el embedding para los documentos
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    return embeddings

def generate_image_embeddings (image_dir: list[str], model_name = "clip-ViT-B-32", device = None, resize_to=(224, 224)) -> tuple[np.ndarray, list[str]]:
    '''
    Input:
    - image_dir
        
        String que menciona el directorio a de las imagenes a generar el embedding, realizado a partir del modelo "clip-ViT-B-32" con un reescalado
    Output:
    - tuple(np.ndarray, [str])
    
        Embedding generado para cada imagen del directorio y su path
    '''
    model = SentenceTransformer(model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    all_embeddings = []
    try:
        image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith('.jpg')]
        imgs = [Image.open(p).convert('RGB').resize(resize_to) for p in image_paths]
    except:
        pass
    all_embeddings = model.encode(
        imgs,
        convert_to_numpy = True,
        device = device,
        show_progress_bar=True
    )
    return np.vstack(all_embeddings), image_paths

def generate_image_embedding (image_path: str, model_name = "clip-ViT-B-32", device = None, resize_to=(224, 224)) -> np.ndarray:
    '''
    Input:
    - image_path
        
        String que menciona el path de la imagen a generar el embedding, realizado a partir del modelo "clip-ViT-B-32" con un reescalado
    Output:
    - np.ndarray
    
        Embedding generado para esa imagen
    '''
    model = SentenceTransformer(model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    try:
        img = Image.open(image_path).convert('RGB').resize(resize_to)
    except:
        pass
    embedding = model.encode(
        img,
        convert_to_numpy=True,
        device=device,
        show_progress_bar=True
    )
    return embedding

def combine_img_embeddings_text_embeddings (txt_embeddings : np.ndarray, img_embeddings : np.ndarray, alpha = .5) ->np.ndarray:
    '''
    Input:
    - txt_embeddings
        
        embeddings de los documentos
    - img_embeddings

        embeddings de las imagenes
    Output:
    - np.ndarray

        combinación normalizada de los embeddings entrantes
    '''
    
    combined_emb = []
    for text_vec, img_vec in zip(txt_embeddings, img_embeddings):
        text_vec = np.array(text_vec)
        img_vec = np.array(img_vec)
        # linalg, normaliza cada vector
        text_vec /= np.linalg.norm(text_vec)
        image_vec /= np.linalg.norm(image_vec)
        # linalg, normalizamos la combinación
        combined = alpha * text_vec + (1 - alpha) * image_vec
        combined /= np.linalg.norm(combined)

        combined_emb.append(combined)
    return combined_emb