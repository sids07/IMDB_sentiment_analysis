## This is an assessment for the DK AI Research Lab Deep Learning Internship.

### Data Creation
- First, we need to prepare data i.e. cleaning, splitting into train, test, etc.

      python preprocess_data.py

- It will take the default value for the following things:
    - data_path = 'data/IMDB Dataset.csv'
    - test_size = 0.1
    - tokenizer_output_path = 'model/tokenizer.json'

- Now you can see three new files under the data folder i.e.
    - data/clean_data.csv
    - data/train.csv
    - data/test.csv
- One new file under the model folder
    - model/tokenizer.json

 - if you want to modify those default values you can do it using the following command:
   
       python data.py --file_path PATH_TO_IMDB_FILE.csv --test_size DESIRED_RATION --tokenizer_output_json PATH_TO_TOKENIZER.json

    - file_path = file for main csv
    - test_size = ratio on which we want to create unseen data
    - tokenizer_output = where we want our tokenizer.json file to be e.g. model/tokenizer.json

Now, we have all our data and word2int JSON.

### Training

- If default values were used for data creation. And, we want to train our model using default parameters. We can do so using the following command.

      python train.py
- For passing custom value we can run commands like:
  
      python train.py --train_file_path data/k/train.csv --word2int_file_path data/k/tokenizer.json --batch_size 32 --num_workers 2 --max_seq 256 --nhead 4 --num_encoder_layer 3 --epochs 1 --score_threshold 0.55 --save_model_path data/k/third_model.pth

    - train_file_path = path to our train.csv
    - word2int_file_path = path to our word2vec json file/ tokenizer.json
    - batch_size = desired batch size for training
    - num_workers = number of workers to be assigned
    - max_seq = max sequence length allowed to process by model
    - nhead = number of heads of multi-head attention
    - num_encoder_layer = number of encoder layers to stack on the model
    - epochs = total epochs for model training
    - score_threshold = score which determines the class value
    - save_model_path = path where we want our model i.e. model.pth

Make sure we know what we have set because some of them are required for inference and evaluation.

### Evaluation of unseen data (OPTIONAL):

- NOTE: We must verify if we have a test.csv file that was not seen during the training procedure.

      python evaluate.py
Here, we set the default path for test.csv as data/test.csv and the model path as model/model.pth, word2int json path as model/tokenizer.json. So, if we want to provide other paths we can do as below.
- For custom values:

      python evaluate.py --model_file_path data/k/third_model.pth --test_file_path data/test.csv --word2int_file_path model/tokenizer.json --batch_size 32

### Inference

- For inferencing on new data:

      python inference.py --text "I love watching movies"

- For the custom path to model_file and word2int file

      python inference.py --model_file_path PATH/TO/model.pth --word2int_file_path PATH/TO/tokenizer.json --text "I love watching movies"

### API Expose:
- Now, since we have everything ready let's expose our API.

        uvicorn api:app --host 0.0.0.0 --port 9099

### UI using Gradio:
- Once our API is on, go to the next terminal and run our UI:

      python gradio_app.py

- This will host UI on 0.0.0.0:9999
- You can try entering your text and get its sentiment. make sure you enter correct host and port on UI for hitting inference API.

### HYPERPARAMETER TUNING:
- For finding best parameters we can do hyperparameter tuning using script:
      python hyperparameter_tuning.py

- We have passed the values we want to test on hyperparameter_config.py. If you want to try new values you can make the modification on config
- NOTE:
      - values of D_MODEL must be divisible by values on NHEAD else error will be thrown as model expects this

- CONFIG Details:
      - BATCH_SIZE
      - D_MODEL = embedding dimension
      - NHEAD = number of heads on multi-layer attention
      - NUM_ENCODER_LAYER = number of transformer encoder to use
      - EPOCHS = training epochs
      - SCORE_THRESHOLD = score to detemine the class labels.
      - N_TRIAL = how many experiments do we want to conduct to find the optimal hyperparameters.

- This will save two files:
      - model/best_hyperparameter.json = will have optimal values for hyperparameters
      - model/best_state_dict = will have optimal weights on the model layers.

- Finally since we have the optimal weights, It can be utilized as:
      from model import TransformersClassificationModel

      model = TransformersClassificationModel(args)

      model.load_state_dict(torch.load("model/best_state_dict", weights_only = True))