TUNING_CONFIG = {
    "BATCH_SIZE":[8,16,32],
    "D_MODEL" : [124,256,380],
    "NHEAD" : [1,2,4],
    "NUM_ENCODER_LAYER" : [1,2,3,4],
    # "EPOCHS" : [5,8,10,13]
    "EPOCHS":[2,3],
    "SCORE_THRESHOLD" : [0.4,0.45,0.5,0.55,0.6],
    "N_TRIALS": 2
}