import os
import numpy as np

from dotenv import load_dotenv
from src.loader_dataset import load_names, get_caption_by_image_name
from src.embeddings import generate_text_embeddings, generate_image_embeddings, combine_img_embeddings_text_embeddings
from src.vector import build_vector_index, search_similar_vectors, save_index, load_index
from src.search import search_unique_products

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')

paths = ['data/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv','data/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv','data/1429_1.csv' ]
img_dir = 'data/img'

df = load_names(paths, img_dir)

embedding = generate_text_embeddings(df['caption'].to_list(), device = 'cuda')
print(type(embedding))

embedding_img, paths = generate_image_embeddings('data/img/', device='cuda')
print(type(embedding_img))

embedding_for_search = np.array(combine_img_embeddings_text_embeddings(embedding, embedding_img))

index = build_vector_index(embedding_for_search)
query = str(input('Ingresa una consulta: '))
df_test = search_unique_products(query, index, df)

print(f"Top {len(df_test)} resultados únicos encontrados:")
for idx, row in df_test.iterrows():
    print(f"- Producto: {row['caption']}")
    print(f"  Categoría: {row['categories']}")
    print(f"  Distancia: {row['distance']:.4f}")
    print("-" * 30)