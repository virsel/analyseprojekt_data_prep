import pandas as pd
import os
import re
from pathlib import Path
from datetime import datetime, timedelta
from ast import literal_eval
from transformers import LlamaTokenizer, AutoTokenizer

dir_path = os.path.dirname(os.path.abspath(__file__))
step2_path = os.path.join(dir_path, "../../output/{}_step2.csv")
out_path = os.path.join(dir_path, "../../output/{}_step3.csv")

# Initialize tokenizer
# tokenizer = LlamaTokenizer.from_pretrained('ChanceFocus/finma-7b-nlp')
tokenizer = AutoTokenizer.from_pretrained("D:\models\T-Systems-onsitecross-en-de-roberta-sentence-transformer")

def load_step2(step2_path):
    df = pd.read_csv(step2_path)
    df['tweets'] = df['tweets'].apply(lambda x: literal_eval(x) if pd.notna(x) else [])
    return df



def encode2tkids(text):
    encoding = tokenizer(
            text,
            max_length=40,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
    
    return encoding['input_ids'].squeeze(0).tolist()

def pad_tweet_list(tweet_list, max_tweets=3):
    # Get the padding token ID (often 0 or the tokenizer's pad token ID)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    
    # Tokenize the tweets
    tokenized_tweets = [encode2tkids(tweet) for tweet in tweet_list]
    
    # If fewer than 3 tweets, pad with a list of pad token IDs
    while len(tokenized_tweets) < max_tweets:
        tokenized_tweets.append([pad_token_id] * 40)  # 40 is your max_length from earlier
    
    # If more than 3 tweets, truncate
    return tokenized_tweets[:max_tweets]


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
        lambda tweet_list: pad_tweet_list(tweet_list)
    )
    
    return df



def main(stock):
    df = load_step2(step2_path.format(stock))
    df_step2 = process_tweet_column(df)
    df_step2.to_csv(out_path.format(stock), index=False)







    


