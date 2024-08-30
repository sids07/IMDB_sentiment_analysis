from torch.utils.data import DataLoader, Dataset, Subset
from pandas import DataFrame
import pandas as pd
import numpy as np
import torch
from typing import List

class IMDBDataset(Dataset):
    
    def __init__(
        self, 
        file_path: str, 
        word2int: dict,
        seq_length: int=256
    )->None:
        
        self.file_path = file_path
        self.seq_length = seq_length
        
        self.dataset = self._load_dataset()
        self.reviews = self.dataset.clean_review.values
        self.word2int = word2int
    
    def _load_dataset(self) -> DataFrame:
        return pd.read_csv(self.file_path)
    
    def return_int_vector(self, review: str) -> List[int]:
        return [self.word2int[word] if word in self.word2int else 1 for word in review.split() ]
    
    def pad_features(self,int_vector:List[int]) -> List[int]:
        features = np.zeros((1, self.seq_length), dtype = int)
        if len(int_vector) <= self.seq_length:
            zeros = list(np.zeros(self.seq_length - len(int_vector)))
            new = int_vector + zeros
        else:
            new = int_vector[: self.seq_length]
        features = np.array(new)
        return features
    
    def encode_labels(self, label:str) -> int:
        if label=="positive":
            return 1
        return 0
    
    def __len__(self)->int:
        return len(self.reviews)
    
    def __getitem__(self, idx: int) -> dict:
        
        df = self.dataset.loc[idx]
        clean_review = df["clean_review"]
        sentiment = df["sentiment"]
        int_vector = self.return_int_vector(review= clean_review)
        padded_features = self.pad_features(int_vector=int_vector)
        encoded_labels = self.encode_labels(label=sentiment)
        return {
            "text": torch.tensor(padded_features, dtype= torch.int32),
            "label": torch.tensor(encoded_labels, dtype = torch.long)
        }  
        
if __name__ == "__main__":
    import json 
    
    with open("model/tokenizer.json","r") as fp:
        word2vec = json.load(fp)
        
    dataset = IMDBDataset(
        file_path = "data/test.csv",
        word2int = word2vec
    )
    
    print(len(dataset))
    print(dataset[1])