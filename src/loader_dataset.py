import pandas as pd

def load_dataset(csv_path : str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[['id','name','categories', 'imageURLs']].drop_duplicates(subset='id')
    return pd.read_csv(csv_path)