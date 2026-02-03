import gradio as gr
import pandas as pd
from PIL import Image
from src.search import retrieve_by_text, retrieve_by_image, retrieve_by_text_and_image
from src.rag import build_prompt, generate_response

def launch_ui(df, index_faiss, gemini_client):
    """
    Lanza la interfaz de Gradio usando los objetos ya cargados en memoria.
    """
    
    def process_request(query_text, query_img, search_type):
        # 1. Lógica de Recuperación
        if search_type == "Texto" and query_text:
            # I ahora es un array plano de índices únicos, ej: [309, 308, 30]
            D, I = retrieve_by_text(query_text, index_faiss, df)
        elif search_type == "Imagen" and query_img:
            D, I = retrieve_by_image(query_img, index_faiss, df)
        elif search_type == "Mixto" and query_text and query_img:
            D, I = retrieve_by_text_and_image(query_text, query_img, index_faiss, df)
        else:
            return "Error en entrada", "", None

        # --- CORRECCIÓN AQUÍ ---
        # Usamos I directamente porque ya es la lista de índices filtrados
        results = df.iloc[I] 
        
        # Verificamos si hay resultados para evitar errores
        if results.empty:
            return "No se encontraron resultados.", "", None

        # Al ser results un DataFrame, ahora sí podemos usar .tolist()
        # Si usaste 'combined_caption' en el merge, asegúrate que la columna exista
        col_name = 'combined_caption' if 'combined_caption' in results.columns else 'caption'
        retrieved_docs = results[col_name].tolist()
        
        # Preparar galería (path, label)
        gallery_items = [(row['image_path'], f"Producto: {row.get('combined_caption', row.get('caption', 'Sin nombre'))}") 
                        for _, row in results.iterrows()]
        
        # 3. RAG: Prompt y Respuesta
        prompt = build_prompt(query_text if query_text else "Consulta visual", retrieved_docs)
        print(prompt)
        response = generate_response(gemini_client, prompt)
        
        return response, prompt, gallery_items
    # Construcción de la Interfaz
    with gr.Blocks(title="Buscador Multimodal RAG", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🤖 Interfaz RAG: Búsqueda de Imágenes y Texto")
        
        with gr.Row():
            with gr.Column(scale=1):
                mode = gr.Radio(["Texto", "Imagen", "Mixto"], label="Modo de búsqueda", value="Texto")
                txt_input = gr.Textbox(label="Escribe tu búsqueda", placeholder="Ej: Amazon Echo Dot...")
                img_input = gr.Image(label="O carga una imagen", type="filepath")
                btn = gr.Button("Ejecutar", variant="primary")
            
            with gr.Column(scale=2):
                with gr.Tab("Respuesta Generativa"):
                    out_res = gr.Markdown(label="Respuesta de Gemini")
                    out_gallery = gr.Gallery(label="Imágenes encontradas", columns=2)
                
                with gr.Tab("Prompt del Sistema"):
                    out_prompt = gr.Code(label="Prompt concatenado", language="markdown")

        btn.click(process_request, [txt_input, img_input, mode], [out_res, out_prompt, out_gallery])

    demo.launch()