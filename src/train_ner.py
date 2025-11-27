import spacy
from spacy.training import Example
from spacy.util import minibatch
import random
import json

def load_training_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def train_ner_model(training_data, n_iter=30):
    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner")
    
    for _, annotations in training_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])
    
    optimizer = nlp.begin_training()
    
    for iteration in range(n_iter):
        random.shuffle(training_data)
        losses = {}
        
        batches = minibatch(training_data, size=8)
        for batch in batches:
            examples = []
            for text, annots in batch:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annots)
                examples.append(example)
            
            nlp.update(examples, drop=0.3, losses=losses)
        
        print(f"Iteration {iteration + 1}/{n_iter}, Loss: {losses['ner']:.2f}")
    
    return nlp

print("Loading training data...")
training_data = load_training_data("training_data.json")
print(f"Loaded {len(training_data)} examples")

print("\nTraining NER model...")
nlp = train_ner_model(training_data, n_iter=30)

print("\nSaving model...")
nlp.to_disk("medical_ner_model")
print("Model saved to 'medical_ner_model/'")