import pandas as pd
import os

from src.preprocessing import clean_text


def load_names(csv_paths: list[str], img_dir: str) -> pd.DataFrame:
    rows = []

    df_name_img = pd.concat(
        [pd.read_csv(p) for p in csv_paths],
        ignore_index=True
    )

    df_name_img = df_name_img[['name', 'imageURLs']].dropna(subset=['imageURLs'])
    df_name_img['name'] = df_name_img['name'].apply(clean_text)

    valid_names = set(df_name_img['name'].values)

    for img_file in os.listdir(img_dir):
        if not img_file.endswith('.jpg'):
            continue

        img_stem = img_file.rsplit("_", 1)[0]

        if img_stem in valid_names:
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