import torch
from transformers import AutoModel, AutoTokenizer

bert_model = AutoModel.from_pretrained("bert-base-cased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

text = "Elon Musk founded SpaceX in 2002."

tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    embeddings = bert_model(**tokens).last_hidden_state

print(embeddings.shape)

model.save_pretrained("ner_chatbot")
tokenizer.save_pretrained("ner_chatbot")
