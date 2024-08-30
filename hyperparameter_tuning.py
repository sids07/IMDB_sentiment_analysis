import torch
import optuna
import json
from trainer import ModelTrainer
from dataloader import IMDBDataLoader
from hyperparameter_config import TUNING_CONFIG

class ModelTunerTrainer(ModelTrainer):
    
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, device, trial):
        
        super().__init__(vocab_size, d_model, nhead, num_encoder_layers, device)
        self.trial = trial
        
        
    def train(self,train_loader, valid_loader, epochs,threshold = 0.5):
        best_valid_accuracy = 0
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
            
            if validation_accuracy > best_valid_accuracy:
                best_valid_accuracy = validation_accuracy
                
                if best_valid_accuracy> best_results["accuracy"]:
                    print("New milestone with accuracy: ", validation_accuracy)
                    best_results["accuracy"] = validation_accuracy
                    best_results["model"] = self.model.state_dict()
            
            self.trial.report(validation_accuracy, epoch)
            self.scheduler.step(validation_loss)
            
            print("Train Loss: ", train_loss)
            
            print("Train Accuracy: ", train_accuracy)
            print("Eval Loss: ", validation_loss)
            print("Eval Accuracy: ", validation_accuracy)
            
        return best_valid_accuracy
    

    
def suggest_hyperparameters(trial, config=TUNING_CONFIG):

    return {
        "batch_size": trial.suggest_categorical("batch_size", config["BATCH_SIZE"]),
        "d_model": trial.suggest_categorical("d_model",config["D_MODEL"]),
        "nhead": trial.suggest_categorical("nhead", config["NHEAD"]),
        "num_encoder_layers": trial.suggest_categorical("num_encoder_layer",config["NUM_ENCODER_LAYER"]),
        "epochs": trial.suggest_categorical("epochs",config["EPOCHS"]),
        "score_threshold": trial.suggest_categorical("score_threshold",config["SCORE_THRESHOLD"])
    }

def objective(
    trial,
    train_file_path="data/train.csv", 
    tokenizer_file_path="model/tokenizer.json", 
    num_workers=2
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    imdb_dataloader = IMDBDataLoader(
        file_path = train_file_path,
        tokenizer_path = tokenizer_file_path
    )
    
    hyperparameters = suggest_hyperparameters(trial)
    print("*"*100)
    print(hyperparameters)
    batch_size = hyperparameters["batch_size"]
    d_model = hyperparameters["d_model"]
    nhead = hyperparameters["nhead"]
    num_encoder_layers = hyperparameters["num_encoder_layers"]
    epochs = hyperparameters["epochs"]
    score_threshold = hyperparameters["score_threshold"]
    
    train_loader, valid_loader, word2int = imdb_dataloader.generate_train_and_valid_data_loader(
        batch_size = batch_size,
        num_worker = num_workers
    )
    
    trainer = ModelTunerTrainer(
        vocab_size = len(word2int),
        d_model = d_model,
        nhead = nhead,
        num_encoder_layers = num_encoder_layers,
        device = device,
        trial= trial
    )
    
    best_val_acc = trainer.train(
        train_loader=train_loader, 
        valid_loader=valid_loader, 
        epochs= epochs, 
        threshold=score_threshold
    )
    
    return best_val_acc

if __name__ == "__main__":
    
    best_results = {
            "accuracy": 0,
            "model":None
        }
    
    n_trials = TUNING_CONFIG["N_TRIALS"]
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=n_trials)
    
    best_hyperparams = study.best_params
    
    with open("model/best_hyperparameter.json","w") as fp:
        
        json.dump(best_hyperparams, fp, indent=4)
        
    torch.save(best_results["model"],"model/best_state_dict")
