from dataloader import IMDBDataLoader
from model import TransformerClassificationModel
from trainer import ModelTrainer
import torch
import argparse

def train(train_file_path, tokenizer_file_path, batch_size, num_workers, d_model, nhead, num_encoder_layer, epochs, score_threshold, save_dir):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    imdb_dataloader = IMDBDataLoader(
        file_path = train_file_path,
        tokenizer_path = tokenizer_file_path
    )
    
    train_loader, valid_loader, word2int = imdb_dataloader.generate_train_and_valid_data_loader(
        batch_size = batch_size,
        num_worker = num_workers
    )
    
    trainer = ModelTrainer(
        vocab_size = len(word2int),
        d_model = d_model,
        nhead = nhead,
        num_encoder_layers = num_encoder_layer,
        device = device
    )
    
    trainer.train(
        train_loader = train_loader,
        valid_loader = valid_loader,
        epochs = epochs,
        threshold = score_threshold
    )
    
    trainer.save(
        output_model_path = save_dir
    )
    

def parse_arguments():
    parser = argparse.ArgumentParser(description="Training Sentiment Analysis model on IMDB Dataset .")
    
    parser.add_argument(
        '--train_file_path', 
        type=str, 
        default='data/train.csv',
        help='Path to the dataset file (CSV format).'
    )
    
    parser.add_argument(
        '--word2int_file_path', 
        type=str, 
        default="model/tokenizer.json",
        help='Path to saved word2int json file'
    )
    
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=32,
        help='Batch Size for creating data loader.'
    )
    
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="No. of workers for data loader creation"
    )
    
    parser.add_argument(
        "--max_seq",
        type = int,
        default = 256,
        help = "Max sequence to be processed by model for training"
    )
    
    parser.add_argument(
        "--nhead",
        type = int,
        default = 4,
        help = "Number of heads for multi-head attention"
    )
    
    parser.add_argument(
        "--num_encoder_layer",
        type = int,
        default = 3,
        help = "Number of transformer encoder layer to model"
    )
    
    parser.add_argument(
        "--epochs",
        type = int,
        default = 5,
        help = "Total number of epochs for model training"
    )
    
    parser.add_argument(
        "--score_threshold",
        type = float,
        default = 0.5,
        help = "Threshold score to determine the class"
    )
    
    parser.add_argument(
        "--save_model_path",
        type = str,
        default = "model/model.pth"
    )
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    args = parse_arguments()
    train_file_path = args.train_file_path
    tokenizer_file_path = args.word2int_file_path
    batch_size = args.batch_size
    num_workers = args.num_workers
    d_model = args.max_seq
    nhead = args.nhead
    num_encoder_layer = args.num_encoder_layer
    epochs = args.epochs
    score_threshold = args.score_threshold
    save_dir = args.save_model_path
    
    train(train_file_path, tokenizer_file_path, batch_size, num_workers, d_model, nhead, num_encoder_layer, epochs, score_threshold, save_dir)