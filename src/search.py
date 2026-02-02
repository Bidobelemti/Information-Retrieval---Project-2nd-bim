import numpy as np
from src.embeddings import generate_image_embeddings, generate_text_embeddings
from src.vector import search_similar_vectors

def search_unique_products(query_text, index, metadata_df, n=3):
    """
    Busca n productos únicos.
    """
    # 1. Generar el embedding de la consulta
    q_emb = generate_text_embeddings(query_text) #
    
    # 2. Oversampling: Pedimos más de n para compensar duplicados (ej: n * 5)
    # k_search debe ser suficiente para encontrar n elementos únicos
    k_search = n * 5 
    D, I = search_similar_vectors(q_emb, index, K=k_search) #
    
    # 3. Mapear índices de FAISS al DataFrame de metadatos
    # I[0] contiene los índices de las filas más cercanas
    results_df = metadata_df.iloc[I[0]].copy()
    results_df['distance'] = D[0]
    
    # 4. Filtrar duplicados por 'caption' (o 'name') y tomar los primeros n
    # keep='first' asegura que mantengas el que tiene la menor distancia/mejor score
    final_results = results_df.drop_duplicates(subset=['caption'], keep='first').head(n)
    
    return final_results