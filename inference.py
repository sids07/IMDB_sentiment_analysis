import torch
import json
import argparse
from utils import prepare_data

class Inference:
    
    def __init__(self, model_path, tokenizer_file_path, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        
        self.model = torch.load(model_path)
        
        with open(tokenizer_file_path,"r") as fp:
            self.word2vec = json.load(fp)
        
        self.device = device
    
    def get_sentiment(self, text):
        
        inp = prepare_data(
            text,
            self.word2vec
        )
        
        with torch.no_grad():
            output = self.model(inp.to(self.device))
        
        result = torch.sigmoid(output)
        
        if result>0.5:
            return "positive"
            
        return "negative"


def parse_arguments():
    
    parser = argparse.ArgumentParser(description = "Inferencing trained model")
    
    parser.add_argument(
        '--model_file_path', 
        type=str, 
        default='model/model.pth',
        help='Path to the trained model.'
    )
    
    parser.add_argument(
        '--word2int_file_path', 
        type=str, 
        default="model/tokenizer.json",
        help='Path to saved word2int json file'
    )
    
    parser.add_argument(
        "--text",
        type = str,
        required = True,
        help = "Text which we want to do sentiment analysis"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    
    args = parse_arguments()
    
    model_path = args.model_file_path
    tokenizer_path = args.word2int_file_path
    
    text = args.text
    
    inf = Inference(
        model_path = model_path,
        tokenizer_file_path = tokenizer_path
    )
    sentiment = inf.get_sentiment(text)
    print(sentiment)