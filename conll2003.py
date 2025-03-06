from datasets import load_dataset
import spacy
from spacy.pipeline import EntityRecognizer
from spacy.tokens import DocBin

dataset = load_dataset("conll2003", trust_remote_code=True)

id2label = dataset["train"].features["ner_tags"].feature.int2str  

def convert_to_spacy_format(dataset):
    spacy_data = []
    for entry in dataset:
        words = entry["tokens"]
        entities = entry["ner_tags"]
        doc = {"text": " ".join(words), "entities": []}
        
        start = 0
        for i, word in enumerate(words):
            entity_label = id2label(entities[i])
            
            if entity_label != "O":  
                doc["entities"].append((start, start + len(word), entity_label))
            
            start += len(word) + 1 
        
        spacy_data.append(doc)
    return spacy_data

train_data = convert_to_spacy_format(dataset["train"])

print(train_data[:2])

import spacy
from spacy.tokens import DocBin

nlp = spacy.blank("en")
ner = nlp.add_pipe("ner")

for label in ["PERSON", "ORG", "LOC", "MISC"]:
    ner.add_label(label)

db = DocBin()

for item in train_data:
    text = item["text"]
    doc = nlp.make_doc(text) 
    
    ents = []
    for start, end, label in item["entities"]:
        try:
            span = doc.char_span(start, end, label=label, alignment_mode="expand")
            if span is not None:
                ents.append(span)
        except Exception as e:
            print(f"Skipping entity due to error: {e}")

    doc.ents = ents
    db.add(doc)

db.to_disk("train.spacy")
