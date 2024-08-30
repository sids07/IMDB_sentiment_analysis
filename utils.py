import torch
import re
import string
import numpy as np

def get_correct_classification_count_in_batch(predictions, labels, score_threshold):
    predictions = torch.sigmoid(predictions)
    correct_prediction_count = 0
    for i, truth in enumerate(labels):
        if predictions[i]<score_threshold and truth<score_threshold:
            correct_prediction_count += 1
        elif predictions[i]>score_threshold and truth>score_threshold:
            correct_prediction_count += 1
            
    return correct_prediction_count

def preprocess(text):
    text = re.sub(r'<[^>]+>', '', text) #HTML TAG removal
    text = ''.join([
            character for character in text \
                if character not in string.punctuation
        ])
    text = text.lower()
    
    return text

def convert_to_num(text, word2int):
    return [word2int[word] if word in word2int else 1 for word in text.split()]

def pad_features(int_vector, seq_length):
    features = np.zeros((1, seq_length), dtype = int)
    if len(int_vector) <= seq_length:
        zeros = list(np.zeros(seq_length - len(int_vector)))
        new = int_vector + zeros
    else:
        new = int_vector[: seq_length]
    features = np.array(new)
    return features
    
def prepare_data(text, word2int, seq_length=256):     
    clean_text = preprocess(text)
    num_vector = convert_to_num(clean_text, word2int)
    pad_vector = pad_features(num_vector, seq_length)
    
    input_tensor = torch.tensor(pad_vector, dtype=torch.int32)
    input_tensor = input_tensor.unsqueeze(0)
    
    return input_tensor