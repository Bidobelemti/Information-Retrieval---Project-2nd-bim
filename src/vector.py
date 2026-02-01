import faiss
import numpy as np

def build_vector_index(embeddings:np.ndarray):
    '''
    Construcción de indice vectorial con FAISS
    
    input: 
    
    - embeddings:
        np.ndarray de embeddings
    Output:
    - Indice de vectorial con FAISS
    '''
    dimension = embeddings.shape[1] 
    index = faiss.IndexFlatIP(dimension)
    index.add(np.array(embeddings))

    return index

def search_similar_vectors(query_emb: np.ndarray, index, K=3):
    '''
    Busqueda de k vectores similares al embedding de la query
    
    Input:
    
    - query_emb
        Embedding de la query

    - index
        Indice de FAISS construido a partir del embedding del corpus

    Output:
    - Documentos e indices más cercanos a la query 
    '''
    query_emb = np.asarray(query_emb, dtype='float32')

    if query_emb.ndim == 1:
        query_emb = query_emb.reshape(1, -1)
    elif query_emb.ndim != 2:
        raise ValueError(f"query_emb debe tener 1 o 2 dimensiones, pero tiene {query_emb.ndim}")

    D, I = index.search(query_emb, K)
    return D, I

def save_index(index, path="index_images.faiss"):
    '''
    Guarda el índice en disco, tiene un path por defecto
    '''
    faiss.write_index(index, path)

def load_index(path="index_images.faiss"):
    '''
    Carga el indice desde el disco, tiene un path por defecto
    '''
    return faiss.read_index(path)