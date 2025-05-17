from emotionClassification.utils.config import Config
from emotionClassification.configs.config import CFGLog
import os
import pickle


class Inferrer:
    def __init__(self):
        self.config = Config.from_json(CFGLog)
        self.saved_path = os.path.join(
            self.config.output.output_path, self.config.output.model_name
        )
        
        self.files = os.listdir(self.config.output.output_path)
        
        with open(self.saved_path, "rb") as f:
            self.vectorizer, self.model = pickle.load(f)

    def preprocess(self, document: str):
        """Converts input document string to embeddings by the trained vectorizer"""
        return self.vectorizer.transform([document])

    def infer(self, document):
        """Converts input document string to embeddings and infer emotion result"""
        document_embeddings = self.preprocess(document)
        predicted_emotion = self.model.predict(document_embeddings)
        print(f"Model in use: {self.saved_path}")
        return predicted_emotion
