from sklearn.model_selection import train_test_split
from pandas import DataFrame
from collections import Counter
import pandas as pd
import re
import string
import json
import argparse

class DataProcessor:
    
    def __init__(self, file_path: str="data/IMDB Dataset.csv") -> None:
        
        self.file_path = file_path
        
    def _load_dataset(self) -> DataFrame:
        return pd.read_csv(self.file_path)
    
    @staticmethod
    def _preprocess_dataset(df: DataFrame) -> DataFrame:
        
        df = df.drop_duplicates(keep='first')
        df["clean_review"] = df["review"].apply(lambda x: DataProcessor._remove_html_and_consecutive_fullstop(x))
        df = df[["clean_review","sentiment"]] 
        df.to_csv("data/clean_main.csv", index=None)
        return df
    
    @staticmethod   
    def _remove_html_and_consecutive_fullstop(text: str) -> str:
        
        text = re.sub(r'<[^>]+>', '', text) #HTML TAG removal
        text = ''.join([
                character for character in text \
                    if character not in string.punctuation
            ])
        return text.lower()
    
    @staticmethod
    def _train_test_split(df: DataFrame, test_size: float, random_state: int)->DataFrame:
        
        train_data, test_data = train_test_split(
            df,
            test_size= test_size,
            random_state= random_state,
            stratify= df["sentiment"]
        )
        print(len(test_data))
        train_data.to_csv("data/train.csv", index=None)
        test_data.to_csv("data/test.csv", index=None)
        
        return True, train_data
    
    @staticmethod
    def _save_word_to_int_json(data: DataFrame, tokenizer_output_path:str) -> bool:
        try:
            counter = Counter((" ".join(data.clean_review.values)).split())
            word_frequency = counter.most_common(n=None)

            word2int ={
                w:i+1 for i, (w,c) in enumerate(word_frequency) if c>5
            }
            word2int["<PAD>"]=0
            # word2int["<UNK>"]=1

            vocab_size = len(word2int)
            
            with open(tokenizer_output_path, "w") as out_file:
        
                json.dump(word2int, out_file, indent=4)
                
            return True
        
        except Exception as e:
            print(f"Error encountered: {e}")
            return False
        
    def process_data(self, test_size: float=0.1, random_state: int=42, tokenizer_output_path:str="model/tokenizer.json") -> bool:
        
        try:
            df = self._load_dataset()
            clean_df = DataProcessor._preprocess_dataset(df)
            
            sucess, train_data = DataProcessor._train_test_split(
                df = clean_df, 
                test_size=test_size, 
                random_state=random_state)
            
            if sucess:
                DataProcessor._save_word_to_int_json(train_data,tokenizer_output_path)
                return True
            return False

        except Exception as e:    
            print(f"Error while processing data: {e}")
            return False
            

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process IMDB dataset for sentiment analysis.")
    
    parser.add_argument(
        '--file_path', 
        type=str, 
        default='data/IMDB Dataset.csv',
        help='Path to the dataset file (CSV format).'
    )
    
    parser.add_argument(
        '--test_size', 
        type=float, 
        default=0.1,
        help='Proportion of the dataset to include in the test split.'
    )
    
    parser.add_argument(
        '--random_state', 
        type=int, 
        default=42,
        help='Random seed for splitting the dataset.'
    )
    
    parser.add_argument(
        "--tokenizer_output_path",
        type=str,
        default="model/tokenizer.json",
        help="Path for saving tokenizer i.e. word2int json"
    )
    args = parser.parse_args()
    return args
            
if __name__=="__main__":
    
    args = parse_arguments()
    data_processor = DataProcessor(
        file_path = args.file_path
    )
    
    prepare_flag = data_processor.process_data(
        test_size = args.test_size,
        random_state = args.random_state,
        tokenizer_output_path = args.tokenizer_output_path
    )
    print(prepare_flag)