import pandas as pd
import numpy as np
import re
import nltk
from nltk.stem import porter
from nltk.corpus import stopwords
import unicodedata

nltk.download('stopwords')
stemmer = porter.PorterStemmer()


# Preprocesamiento de texto

def clean_text(doc):
    """
    Limpia y normaliza texto: conversión a minúsculas y eliminación de caracteres no alfabéticos.
    ### Input:
             doc: string
    ### Output:
             doc: string

    """
    doc = unicodedata.normalize('NFKD', doc).encode('ascii', 'ignore').decode('utf-8').lower()
    
    doc = re.sub(r'[^a-z\s]', ' ', doc)
    
    doc = re.sub(r'\s+', ' ', doc).strip()
    
    return doc

def remove_stopwords(doc):
    """
    Elimina stopwords del documento. Toma la lista de stop_words en inles de NLTK
    ### Input:
             doc: string
    ### Output:
             doc: string
    """
    stop_words = set(stopwords.words('english'))
    tokens = doc.split()
    return ' '.join(word for word in tokens if word not in stop_words)

def stemming(doc):
    """
    Aplica stemming a todas las palabras del documento usando PorterStemmer.
    ### Input:
             doc: string
    ### Output:
             doc: string
    """
    tokens = doc.split()
    return ' '.join(stemmer.stem(word) for word in tokens)

def filter_tokens(doc):
    """
    Filtra tokens por longitud, patrones válidos y estructura de palabras.
    ### Input:
             doc: string
    ### Output:
             doc: string
    """
    tokens = doc.split()
    
    valid_tokens = [
        tok for tok in tokens
        if 2 <= len(tok) <= 20  # Longitud válida
        and not re.search(r'(.)\1{2,}', tok)  # Sin 3+ caracteres repetidos
        and sum(c in 'aeiou' for c in tok) > 0  # Al menos una vocal
        and not re.search(r'[^aeiou]{5,}', tok)  # Máximo 4 consonantes consecutivas
    ]
    
    return ' '.join(valid_tokens)

def preprocess_documents(documents, return_type='df'):
    """
    Preprocesa una lista de documentos usando el pipeline propio.
    """

    df = pd.DataFrame(documents, columns=['document'])

    df['clean'] = df['document'].apply(clean_text)
    df['no_stopwords'] = df['clean'].apply(remove_stopwords)
    df['stemmed'] = df['no_stopwords'].apply(stemming)
    df['filtered'] = df['stemmed'].apply(filter_tokens)

    if return_type == 'tokens':
        return df['filtered'].apply(lambda x: x.split()).tolist()
    else:
        return df[['document', 'filtered']].rename(
            columns={'filtered': 'prep_doc'}
        )
    
def preprocess_both(text: str) -> tuple[str, list[str]]:
    df = preprocess_documents([text])
    clean_text_out = df['prep_doc'].iloc[0]
    tokens = clean_text_out.split()
    return clean_text_out, tokens

def merge_captions_by_image(
    df,
    image_col='image_path',
    caption_col='caption'
):
    merged_df = (
        df.groupby(image_col)[caption_col]
        .apply(lambda caps: ' '.join(caps))
        .reset_index()
        .rename(columns={caption_col: 'combined_caption'})
    )
    return merged_df
