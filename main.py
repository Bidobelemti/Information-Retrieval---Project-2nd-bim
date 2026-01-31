from src.loader_dataset import load_dataset
path = 'data/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv'
df = load_dataset(path)
print(len(df))