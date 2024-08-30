import torch
import torch.nn as nn
from dataloader import IMDBDataLoader
from tqdm import tqdm
from utils import get_correct_classification_count_in_batch
import argparse


def evaluate(
    model, 
    test_data, 
    loss_function, 
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    
    model.eval()
    
    test_loss = 0.0
    test_correct = 0
    total_count = 0
    counter = 0
    
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
            
            counter+=1
            predictions =torch.squeeze(model(data["text"].to(device)),-1)
            
            labels = torch.tensor(data["label"], dtype=torch.float32).to(device)
            
            test_loss = loss_function(predictions, labels)

            correct = get_correct_classification_count_in_batch(predictions, labels, score_threshold=0.5)

            test_correct += correct
            total_count += len(labels)
            test_loss += test_loss.item()
            
    total_loss = test_loss/counter
    
    total_accuracy = (test_correct/total_count)*100
    
    return total_loss, total_accuracy   

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process IMDB dataset for sentiment analysis.")
    
    parser.add_argument(
        '--model_file_path', 
        type=str, 
        default='model/model.pth',
        help='Path to the trained model.'
    )
    
    
    parser.add_argument(
        '--test_file_path', 
        type=str, 
        default='data/test.csv',
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
        help='Proportion of the dataset to include in the test split.'
    )
    
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="No. of workers for data loader creation"
    )
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    args = parse_arguments()
    model_path = args.model_file_path
    test_file_path = args.test_file_path
    tokenizer_path = args.word2int_file_path
    batch_size = args.batch_size
    num_workers = args.num_workers
    
    model = torch.load(model_path)     
    
    dataloader = IMDBDataLoader(
        file_path = test_file_path,
        tokenizer_path = tokenizer_path
    )
    
    test_loader = dataloader.create_data_loader(
        data = dataloader.train_data,
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers
    )
    
    criterion = nn.BCEWithLogitsLoss()

    loss, acc = evaluate(
        model = model,
        test_data = test_loader,
        loss_function = criterion
    )
    
    print(loss, acc)
    
    