CFGLog = {
    "data": {
        "path": "./data/Emotion_joy_anger.csv",
        "x": "Comment",
        "y": "Emotion",
        "test_size": 0.2,
        "ngram_range": (1, 2),
        "random_state": 42,
    },
    "train": {
        "solver": "liblinear",
        "penalty": "l1",
        "max_iter": 1000,
        "random_state": 42,
        "C": 1.0,
    },
    "output": {
        "output_path": "./data/exported_models/",
        "model_name": "2024-09-16_18-02-09_LogReg.pickle",
    },
}
