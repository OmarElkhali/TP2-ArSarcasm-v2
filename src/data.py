"""Séparation et exclusion des collisions, sans utiliser les labels du test."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from .preprocess import duplicate_key

def load_splits(root):
    root=Path(root)
    full=pd.read_csv(root/'data/raw/training_data.csv')
    full['id']=range(len(full)); full['y']=full.sarcasm.astype(int)
    test_text=pd.read_csv(root/'data/raw/testing_data.csv',usecols=['tweet'])
    train,val=train_test_split(full,test_size=.1,random_state=42,stratify=full.sarcasm)
    train,val=train.copy(),val.copy()
    for f in [train,val,test_text]: f['key']=f.tweet.map(duplicate_key)
    conflicts=set(train.groupby('key').y.nunique().loc[lambda s:s>1].index)
    forbidden=set(val.key)|set(test_text.key)|conflicts
    train=train.loc[~train.key.isin(forbidden)&train.key.ne('')].drop_duplicates('key')
    assert set(train.key).isdisjoint(val.key) and set(train.key).isdisjoint(test_text.key)
    return train.reset_index(drop=True),val.reset_index(drop=True),test_text
