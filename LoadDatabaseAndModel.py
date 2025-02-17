import pandas as pd
import spacy
csv_path = 'C:\\Users\\fgmz2\\.cache\\kagglehub\\datasets\\sachinkumar62\\movies-details\\versions\\1\\movies.csv'
nlp = spacy.load("en_core_web_sm")
df = pd.read_csv(csv_path)
print(df.head())
