import pandas as pd
import os
import re
from pathlib import Path
from datetime import datetime, timedelta
from config import stock
from ast import literal_eval
from transformers import LlamaTokenizer

dir_path = os.path.dirname(os.path.abspath(__file__))
step2_path = os.path.join(dir_path, f"../../output/{stock}_step2.csv")
out_path = os.path.join(dir_path, f"../../output/{stock}_step3.csv")

# Initialize tokenizer
tokenizer = LlamaTokenizer.from_pretrained('ChanceFocus/finma-7b-nlp')

def load_step2(step2_path):
    df = pd.read_csv(step2_path)
    df['tweets'] = df['tweets'].apply(lambda x: literal_eval(x) if pd.notna(x) else [])
    return df

df = load_step2(step2_path)

def encode2tkids(text):
    encoding = tokenizer(
            text,
            max_length=40,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
    
    return encoding['input_ids'].squeeze(0).tolist()


def process_tweet_column(df):
    """
    Processes the 'tweets' column of a DataFrame, masking elements in each tweet.
    
    Parameters:
    df (pd.DataFrame): DataFrame with a 'tweets' column containing lists of tweet texts
    
    Returns:
    pd.DataFrame: DataFrame with processed tweets
    """
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Process each list of tweets
    df['tweets'] = df['tweets'].apply(
        lambda tweet_list: [encode2tkids(tweet) for tweet in tweet_list]
    )
    
    return df

df_step2 = process_tweet_column(df)
df_step2.to_csv(out_path, index=False)







    


