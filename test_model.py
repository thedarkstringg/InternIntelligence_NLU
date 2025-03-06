import spacy
nlp_ner = spacy.load("./output/model-best")

text = "Apple was founded by Steve Jobs in California."
doc = nlp_ner(text)

for ent in doc.ents:
    print(f"{ent.text} -> {ent.label_}")



