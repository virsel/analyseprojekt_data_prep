import pandas as pd
import os
import re
from pathlib import Path
from datetime import datetime, timedelta
from ast import literal_eval
from transformers import AutoTokenizer, AutoModel
import ast
import numpy as np
import torch
from pathlib import Path
import pickle

# Get the directory of the executing Python file
script_dir = Path(__file__).parent.resolve()

data_path = str(script_dir / "../../output/{}_step3.csv")
out_table_path = str(script_dir / "../../output/{}_embtable.pt")
out_mapping_path = str(script_dir / "../../output/{}_embtable_mapping.pkl")

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("D:\models\T-Systems-onsitecross-en-de-roberta-sentence-transformer")
base_model = AutoModel.from_pretrained("D:\models\T-Systems-onsitecross-en-de-roberta-sentence-transformer")

def load_data(path):
    df = pd.read_csv(path)
    df['tweets'] = df['tweets'].apply(lambda x: literal_eval(x) if pd.notna(x) else [])
    return df


def main(df, file_name):
    # 14900 unique tk ids
    tokenids = list(set(np.array(df['tweets'].to_list()).flatten()))
    table, mapping = create_token_embedding_table(tokenids)
    torch.save(table, out_table_path.format(file_name))
    # Save to a pkl file
    with open(out_mapping_path.format(file_name), 'wb') as file:
        pickle.dump(mapping, file)
        
# Create embedding lookup table
def create_token_embedding_table(tokenids):
    # Get the vocabulary size
    vocab_size = len(tokenids)
    
    # Initialize embedding table
    embedding_dim = base_model.config.hidden_size
    token_embeddings = torch.zeros(vocab_size, embedding_dim)
    
    token_mapping = {}
    
    # Compute embeddings for each token
    with torch.no_grad():
        for i in range(vocab_size):
            token_id = tokenids[i]
            token_mapping[token_id] = i
            
            # Create input tensor for single token
            input_ids = torch.tensor([[token_id]])
            
            # Get token embedding (using the first token's embedding from last hidden state)
            token_emb = base_model(input_ids).last_hidden_state[0, 0, :]
            token_embeddings[i] = token_emb
    
    return token_embeddings, token_mapping


if __name__ == '__main__':
    file_name = 'AMZN_mixed'
    df = load_data(data_path.format(file_name))
    main(df, file_name)







    


