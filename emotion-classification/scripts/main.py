"""main executing script to train, export and infer"""

from emotionClassification.configs.config import CFGLog
from emotionClassification.model.emotion_model import EmotionLogisticRegression
from emotionClassification.executor.inferrer import Inferrer

document_angry = (
    r"i finally fell asleep feeling angry useless and still full of anxiety"
)

document_joy = r"i feel so happy and grateful for everything in my life"


def run():
    """Builds model, loads data, trains, and evaluates"""
    # # Load data and train model
    # mymodel = EmotionLogisticRegression(CFGLog)
    # mymodel.load_data()
    # mymodel.build()
    # mymodel.train()
    # # Predicts results on test set
    # mymodel.evaluate()
    # # Export to pickle
    # mymodel.export_model()

    # Inference
    inferrer = Inferrer()
    print(inferrer.infer(document_angry))


if __name__ == "__main__":
    run()
