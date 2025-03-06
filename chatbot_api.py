from fastapi import FastAPI
from pydantic import BaseModel
import spacy

nlp = spacy.load("C:/Users/fgmz2/OneDrive/Documents/GitHub/InternIntelligence_NLU/output/model-best")

app = FastAPI()

class UserMessage(BaseModel):
    text: str

@app.post("/chat")
def process_message(message: UserMessage):
    
    doc = nlp(message.text)
    
    
    structured_response = {"Person": [], "Location": [], "Organization": []}
    
    for ent in doc.ents:
        if ent.label_ == "PERSON" or "PER" in ent.label_:
            structured_response["Person"].append(ent.text)
        elif ent.label_ == "LOC" or "GPE" in ent.label_ or "LOC" in ent.label_:
            structured_response["Location"].append(ent.text)
        elif ent.label_ == "ORG" or "ORG" in ent.label_:
            structured_response["Organization"].append(ent.text)
    
    return {"message": message.text, "entities": structured_response}
