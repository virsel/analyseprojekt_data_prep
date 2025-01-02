
from transformers import BertTokenizer, BertForSequenceClassification, AutoModel
from transformers import pipeline
import numpy as np
import torch

# Initialisierung des Sprachmodells
model = BertForSequenceClassification.from_pretrained("D:\models\FinancialBERT-Sentiment-Analysis")
tokenizer = BertTokenizer.from_pretrained("D:\models\FinancialBERT-Sentiment-Analysis")
# Initialisierung der Klassifikations-Pipeline
nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Funktion zur Extraktion von Sentiment Metriken über eine Tweet-Liste
def comp_sent(texts):
    sent_res = []
    for text in texts:
        res = nlp(text)
        # Klassifikation-Resultat für einen Text
        # Format: [{'label': <Wert>, 'score': <Wert>}]
        sentiment = res[0]
        
        # Umwandlung zu Format: [positive_score, negative_score]
        res_formatted = [float(sentiment['label'] == 'positive') * sentiment['score'], 
             float(sentiment['label'] == 'negative') * sentiment['score']
             ]
        # Hinzufügen des Resultats zur Liste
        sent_res.append(res_formatted)
    
    # Berechnung des Durchschnitts der "Tuple"
    res = list(np.mean(sent_res, axis=0)) if len(sent_res) > 0 else [0, 0]
        
    # Rückgabe des Durchschnitts über alle Tweets an diesem Tag als Dictionary
    return {
        'positive': res[0],
        'negative': res[1]
    }
    
def comp_emb(tweet_list):
    # Verbinden der Tweets mit Special-Token zu einem Text
    text = ' [SEP] '.join(tweet_list)
    # Klassifikations-Sonderzeichen voranstellen
    text = '[CLS] ' + text
    
    # Text zu Token-IDs umwandeln
    inputs = tokenizer(
        text, 
        padding=True, 
        truncation=True, 
        return_tensors="pt"
    )
    
    # Gesamt-Embedding berechnen
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        embeddings = outputs.hidden_states[-1][:, 0, :].reshape(-1).tolist()  
    
    return embeddings

