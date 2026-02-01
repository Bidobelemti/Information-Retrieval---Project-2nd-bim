import pandas as pd
import os

from src.preprocessing import clean_text


def load_names(csv_paths: list[str], img_dir: str)->pd.DataFrame:
    '''
    ## Input
        - csv_paths: list[str]

            Es una lista que contiene los paths de cada csv a usar
        - img_dir: str

            Es el path del directorio que contiene las imagenes
    ## Output
        - pd.DataFrame
            Retorna un DataFrame con columnas 'image_path', 'caption' (nombre de la imagen) 
    '''
    rows = []

    df_name_img = pd.concat([pd.read_csv(p) for p in csv_paths],ignore_index=True)
    df_name_img = df_name_img[['name', 'imageURLs']].drop_duplicates(subset = 'name').dropna(subset = 'imageURLs')
    df_name_img['name'] = df_name_img['name'].apply(clean_text)
    for img_file in os.listdir(img_dir):
        if not img_file.endswith('.jpg'):
            continue

        img_stem = img_file.rsplit("_", 1)[0]

        if img_stem in df_name_img['name'].values:
            rows.append({
                'image_path': os.path.join(img_dir, img_file).replace('\\', '/'),
                'caption': img_stem
            })

    return pd.DataFrame(rows)

def get_caption_by_image_name(img_name: str, df: pd.DataFrame)->str:
    '''
    ## Input
        - img_name : str

            El nombre del archivo o path del archivo que buscamos.
        - df : pd.DataFrame

            El DataFrame que contiene ('image_path', 'caption')
    
    ## Output
        - str
            El nombre del archivo que estamos buscando
    '''
    try:
        return df.loc[df['image_path'].str.contains(img_name), 'caption'].values[0]
    except:
        return 'No se ha encontrado.'