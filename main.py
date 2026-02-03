import os
import numpy as np
import pandas as pd
import google.generativeai as genai

from dotenv import load_dotenv

from src.loader_dataset import load_names, get_caption_by_image_name
from src.embeddings import generate_text_embeddings, generate_image_embeddings, combine_img_embeddings_text_embeddings
from src.vector import build_vector_index, search_similar_vectors, save_index, load_index
from src.preprocessing import merge_captions_by_image, preprocess_documents

from src.UI.interface import launch_ui

# Carga de API de google para RAG
load_dotenv()
API_KEY = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=API_KEY)
client = genai.GenerativeModel('gemini-3-flash-preview')

TOP_K = 3

paths = ['data/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv','data/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv','data/1429_1.csv' ]
img_dir = 'data/img'

df_raw = load_names(paths, img_dir)
df = merge_captions_by_image(df_raw, image_col='image_path', caption_col='caption')

# Obteniendo los embeddings del corpus text
df_processed = preprocess_documents(df['combined_caption'].tolist())
df['prep_doc'] = df_processed['prep_doc']
text_embeddings = generate_text_embeddings(df['prep_doc'].tolist(), device='cuda')
df['text_embedding'] = [vec for vec in text_embeddings]

# Obteniendo los embeddings del corpus imagen
img_embeddings, img_paths = generate_image_embeddings(img_dir, device = 'cuda')
df_img = pd.DataFrame({
    'image_path' : [path for path in img_paths],
    'img_embedding' : list(img_embeddings)
})
df = df.merge(df_img, on='image_path', how = 'left')

# Obteniendo indice generado a partir de texto e imagenes en FAISS
combined_emb = combine_img_embeddings_text_embeddings(np.array(df['text_embedding']), np.array(df['img_embedding']))

index_faiss = build_vector_index(np.array(combined_emb))
save_index(index_faiss)
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
#        print(f"Modelo disponible: {m.name}")
        pass
launch_ui(df, index_faiss, client)