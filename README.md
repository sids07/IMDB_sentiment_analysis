This is an assesment for DK AI Research Lab Deep Learning Internship.

To replicate this first run data.py to preprocess, split data into train, test and finally create vocab and word2int json.

you can simply run:

    python data.py

It will assume following things:
    data_path = 'data/IMDB Dataset.csv'
    test_size = 0.1
    tokenizer_output_path = 'model/tokenizer.json'

Now you can see three new files under data folder i.e. clean_data.csv, train.csv, test.csv and one new file under model folder i.e. tokenizer.json which is word2vec json.

Else if you want to change this three parameters that you can run:
    python data.py --file_path PATH_TO_IMDB_FILE.csv --test_size DESIRED_RATION --tokenizer_output_json PATH_TO_TOKENIZER.json

- file_path = file for main csv
- test_size = ratio on which we want to create unseen data
- tokenizer_output = where we want our tokenizer.json file to be e.g. model/tokenizer.json

Now, we have all our data and word2int json.

We can start training:

If we have not provided any args and used default python data.py
    python train.py

If we want custom variables:
    python train.py --train_file_path data/k/train.csv --word2int_file_path data/k/tokenizer.json --batch_size 32 --num_workers 2 --max_seq 256 --nhead 4 --num_encoder_layer 3 --epochs 1 --score_threshold 0.55 --save_model_path data/k/third_model.pth

- train_file_path = path to our train.csv
- word2int_file_path = path to our word2vec json file/ tokenizer.json
- batch_size = desired batch size for training
- num_workers = number of workers to be assigned
- max_seq = max sequence length allowed to process by model
- nhead = number of heads of multi-head attention
- num_encoder_layer = number of encoder layer to stack on model
- epochs = total epochs for model training
- score_threshold = score which determines the class value
- save_model_path = path where we want our model i.e. model.pth

Make sure we know what we have set because some of them are required for inference and evaluation.

Now, we want to evaluate our model on unseen data.
    python evaluate.py

For custom values:
    python evaluate.py --model_file_path data/k/third_model.pth --test_file_path data/test.csv --word2int_file_path model/tokenizer.json --batch_size 32

Here we need to make sure model_file_path, word2int_file_path matches the path given during training.


For inferencing on new data:
    python inference.py --text "I love watching movies"

For custom path to model_file and word2int file
    python inference.py --model_file_path PATH/TO/model.pth --word2int_file_path PATH/TO/tokenizer.json --text "I love watching movies"


Now, since we have everything ready lets expose our API.

    uvicorn api:app --host 0.0.0.0 --port 9099

Once our API is on, go to next terminal and run our UI:

    python gradio_app.py

This will host UI on 0.0.0.0:9999

You can try entering your text and get its sentiment. make sure you enter correct host and port on UI for hitting inference API.

