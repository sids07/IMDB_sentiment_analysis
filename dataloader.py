from data import IMDBDataset
from torch.utils.data import DataLoader, Subset
import json
import torch
from typing import Tuple

class IMDBDataLoader:
    
    def __init__(
        self, 
        file_path: str, 
        tokenizer_path: str
    ):    
        with open(tokenizer_path,"r") as fp:
            self.word2int = json.load(fp)
            
        self.train_data = IMDBDataset(
            file_path = file_path, 
            word2int = self.word2int
        )
    
    @staticmethod
    def _split_into_train_and_valid_set(
        train_data: IMDBDataset,
        valid_ratio: float
    ) -> Tuple[DataLoader, DataLoader]:
        
        valid_size = int(len(train_data) * valid_ratio)
        print(valid_size)
        indices = torch.randperm(len(train_data)).tolist()

        train = Subset(train_data, indices[:-valid_size])
        valid = Subset(train_data, indices[-valid_size:])
        
        return train, valid
    
    def generate_train_and_valid_data_loader(
        self,
        batch_size: int=32, 
        num_worker: int=2, 
        validation_ratio: float=0.1):
        
        train_set, valid_set = IMDBDataLoader._split_into_train_and_valid_set(
            train_data = self.train_data,
            valid_ratio = validation_ratio
        )
        
        train_loader = self.create_data_loader(
            data = train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers= num_worker
        )
        
        valid_loader = self.create_data_loader(
            data= valid_set, 
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_worker
        )
        
        return train_loader, valid_loader, self.word2int
        
    def create_data_loader(self, data, batch_size, num_workers, shuffle):
        
        return DataLoader(
            data,
            batch_size = batch_size,
            shuffle = shuffle,
            num_workers = num_workers
        )

if __name__ == "__main__":
    
    train_path = "data/train.csv"
    tokenizer_path = "model/tokenizer.json"
    
    imdb_dataloader = IMDBDataLoader(
        file_path = train_path,
        tokenizer_path = tokenizer_path
    )
    
    batch_size = 32
    num_workers = 2
    
    train, valid = imdb_dataloader.create_data_loader(
        batch_size = batch_size,
        num_worker = num_workers
    )
    
    first_batch = next(iter(train))

    # Print the first item in the batch
    print(first_batch)
    
    first_batch = next(iter(valid))

    # Print the first item in the batch
    print(first_batch)    
    
        
        