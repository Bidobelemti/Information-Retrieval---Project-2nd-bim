import numpy as np
from src.embeddings import generate_image_embeddings, generate_text_embeddings, generate_image_embedding
from src.vector import search_similar_vectors

def retrieve_by_text(query_text, index, k = 3):
    text_emb = emb_text_query(query_text)
    return search_similar_vectors(text_emb, index, k*5)

def retrieve_by_image(query_image_path, index, k = 3):
    image_emb = emb_image_query(query_image_path)
    return search_similar_vectors(image_emb, index, k*5)

def retrieve_by_text_and_image(query_text, query_image_path, index, k = 3):
    text_emb = emb_text_query(query_text)
    image_emb = emb_image_query(query_image_path)

    text_emb = text_emb / np.linalg.norm(text_emb)
    image_emb = image_emb / np.linalg.norm(image_emb)

    if len(query_text.split()) <= 4:
        alpha = 0.3  
    else:
        alpha = 0.5  

    combined_emb = alpha * text_emb + (1 - alpha) * image_emb

    return search_similar_vectors(combined_emb, index, k*5)

def emb_text_query(query):
    txt_embeddings = generate_text_embeddings([query])
    print(f"Texto: {query}")
    print(f"Shape del embedding: {txt_embeddings.shape}") 
    return txt_embeddings[0]

def emb_image_query(query_path):
    img_embeddings = generate_image_embedding(query_path)
    return img_embeddings