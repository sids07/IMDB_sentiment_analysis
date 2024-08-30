from model import TransformerClassificationModel
from dataloader import IMDBDataLoader
from utils import get_correct_classification_count_in_batch
from tqdm import tqdm
import torch.nn as nn
import torch

class ModelTrainer:
    
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, device, dim_feedforward=2048, dropout=0.1):
        
        self.model = TransformerClassificationModel(
            vocab_size=vocab_size, 
            d_model=d_model, 
            nhead=nhead, 
            num_encoder_layers=num_encoder_layers, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout
        ).to(device)
        
        self.criterion = nn.BCEWithLogitsLoss()

        lr = 1e-4

        self.optimizer = torch.optim.Adam(
            (p for p in self.model.parameters() if p.requires_grad),
            lr= lr
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        self.device = device
    
    def _train(self, train_loader, threshold):
        
        self.model.train()
        
        train_loss = 0
        train_correct_count = 0
        total_count = 0
        counter = 0
        
        for idx, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            
            counter+=1
            self.optimizer.zero_grad()
            
            predictions = torch.squeeze(self.model(batch["text"].to(self.device)),-1)
            labels = torch.tensor(batch["label"], dtype=torch.float32).to(self.device)
            
            loss = self.criterion(predictions, labels)
            
            correct_count = get_correct_classification_count_in_batch(
                predictions = predictions,
                labels = labels,
                score_threshold = threshold
            )
            
            train_correct_count += correct_count
            total_count += len(labels)
            train_loss += loss.item()

            loss.backward()

            self.optimizer.step()
        
        total_loss = train_loss/counter
        accuracy = (train_correct_count/total_count) * 100
        
        return total_loss, accuracy
    
    def _eval(self, valid_loader, threshold):
        self.model.eval()
        
        with torch.no_grad():
            eval_loss = 0
            eval_correct_count = 0
            total_count = 0
            counter = 0
            
            for idx, batch in tqdm(enumerate(valid_loader), total=len(valid_loader)):
                
                counter += 1
                
                predictions = torch.squeeze(self.model(batch["text"].to(self.device)),-1)
                labels = torch.tensor(batch["label"], dtype=torch.float32).to(self.device)
                
                loss = self.criterion(predictions, labels)
                
                correct_count = get_correct_classification_count_in_batch(
                    predictions = predictions,
                    labels = labels,
                    score_threshold = threshold
                )
                
                eval_correct_count += correct_count
                total_count += len(labels)
                eval_loss += loss.item()
        
        total_loss = eval_loss/counter
        accuracy = (eval_correct_count/total_count) * 100
        
        return total_loss, accuracy
    
    def train(self,train_loader, valid_loader, epochs, threshold = 0.5):
        
        print("Training Starting")
        for epoch in range(epochs):
            print(f"{epoch=}")
            train_loss, train_accuracy = self._train(
                train_loader = train_loader,
                threshold = threshold
            )
            
            validation_loss, validation_accuracy = self._eval(
                valid_loader = valid_loader,
                threshold = threshold
            )
            
            self.scheduler.step(validation_loss)
            
            print("Train Loss: ", train_loss)
            
            print("Train Accuracy: ", train_accuracy)
            print("Eval Loss: ", validation_loss)
            print("Eval Accuracy: ", validation_accuracy)
            
    
    def save(self, output_model_path):
        
        torch.save(
            self.model,
            output_model_path
        )