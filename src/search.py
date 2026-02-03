import numpy as np
from src.embeddings import generate_text_embeddings, generate_image_embedding
from src.vector import search_similar_vectors
from sentence_transformers import CrossEncoder

def retrieve_by_text(query_text, index, metadata_df, k=3):
    """Búsqueda por texto con sobremuestreo y reranking."""
    text_emb = emb_text_query(query_text)
    
    # 1. Aumentamos el oversampling a k*20 para asegurar diversidad
    # Si tienes muchas fotos del mismo producto, k*10 podría no ser suficiente
    D, I = search_similar_vectors(text_emb, index, k * 20)
    
    # 2. Pasamos el 'k' original como objetivo final
    return _process_results(query_text, D, I, metadata_df, k)

def retrieve_by_image(query_image_path, index, metadata_df, k=3):
    """Búsqueda por imagen con sobremuestreo."""
    image_emb = emb_image_query(query_image_path)
    D, I = search_similar_vectors(image_emb, index, k * 20)
    
    return _process_results(None, D, I, metadata_df, k)

def retrieve_by_text_and_image(query_text, query_image_path, index, metadata_df, k=3):
    """Búsqueda multimodal combinada."""
    text_emb = emb_text_query(query_text)
    image_emb = emb_image_query(query_image_path)

    text_emb = text_emb / np.linalg.norm(text_emb)
    image_emb = image_emb / np.linalg.norm(image_emb)

    alpha = 0.3 if len(query_text.split()) <= 4 else 0.5
    combined_emb = alpha * text_emb + (1 - alpha) * image_emb

    D, I = search_similar_vectors(combined_emb, index, k * 20)
    return _process_results(query_text, D, I, metadata_df, k)

# --- Funciones de Apoyo ---

def _process_results(query_text, D, I, metadata_df, target_k):
    """
    Filtra duplicados reales y aplica un Reranker para ordenar los mejores candidatos.
    """
    # 1. Mapeo a Metadata
    res_df = metadata_df.iloc[I[0]].copy()
    res_df['faiss_score'] = D[0]
    
    # 2. Detección de duplicados
    # Usamos 'caption' porque es el ID que identifica al producto (nombre del archivo sin _1, _2)
    # Evita usar 'prep_doc' si las imágenes tienen descripciones ligeramente distintas.
    if 'caption' in res_df.columns:
        id_col = 'caption'
    else:
        id_col = res_df.columns[0] # Fallback
        
    unique_df = res_df.drop_duplicates(subset=[id_col], keep='first').copy()
    
    # 3. Reranking (Cross-Encoder)
    # Solo si hay una consulta de texto y tenemos suficientes candidatos para comparar
    if query_text and len(unique_df) > 1:
        try:
            # Modelo ligero pero potente para re-ordenar
            reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            
            # Preparamos el texto descriptivo: Nombre Completo + Categorías
            pairs = []
            for _, row in unique_df.iterrows():
                # Combinamos columnas de interés para que el reranker decida
                desc = f"{row.get('full_name', '')} {row.get('categories', '')}"
                pairs.append([query_text, desc])
            
            scores = reranker.predict(pairs)
            unique_df['rerank_score'] = scores
            
            # Ordenamos por la relevancia real del Cross-Encoder
            unique_df = unique_df.sort_values(by='rerank_score', ascending=False)
        except Exception as e:
            print(f"Aviso: Falló el Reranking, usando orden de FAISS. Error: {e}")

    # 4. Retorno del Top K real
    final_df = unique_df.head(target_k)
    
    return final_df['faiss_score'].values, final_df.index.values

def emb_text_query(query):
    return generate_text_embeddings([query])[0]

def emb_image_query(query_path):
    return generate_image_embedding(query_path)