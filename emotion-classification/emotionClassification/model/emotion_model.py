"""Logistic Regression class"""

from sklearn.linear_model import LogisticRegression
from datetime import datetime
from .base_model import BaseModel
from emotionClassification.dataloader.dataloader import DataLoader
from emotionClassification.executor.trainer import LogisticRegressionTrainer
from emotionClassification.utils.postprocessing import ModelSaving


class EmotionLogisticRegression(BaseModel):
    """Logistic Regression model class"""

    def __init__(self, config):
        super().__init__(config)
        self.name = "LogisticRegression"

    def load_data(self):
        """Loads and preprocesses data to inputs accepted by the model"""
        self.dataset = DataLoader().load_data(self.config.data)
        (
            self.x,
            self.y,
            self.x_train,
            self.x_test,
            self.y_train,
            self.y_test,
            self.X_train,
            self.X_test,
            self.Y_train,
            self.Y_test,
            self.vectorizer,
        ) = DataLoader().preprocess_data(
            data_config=self.config.data, dataset=self.dataset
        )

    def build(self):
        """Builds the model"""
        self.model = LogisticRegression()
        print("Logistic Regression model built")

    def train(self):
        """Compiles and trains the model with configured training hyper-parameters"""
        print("Set training hyper parameters")
        self.model = LogisticRegression(
            solver=self.config.train.solver,
            max_iter=self.config.train.max_iter,
            random_state=self.config.train.random_state,
            C=self.config.train.C,
            penalty=self.config.train.penalty,
        )
        print("Training has started")
        start_time = datetime.now()
        trainer = LogisticRegressionTrainer(
            model=self.model, X_train=self.X_train, Y_train=self.Y_train
        )
        trainer.train()
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        print(f'Training is completed in {"{:.2f}".format(training_time)} seconds')

    def evaluate(self):
        """Predicts resuts for the test dataset"""
        self.Y_test_pred = self.model.predict(self.X_test)
        self.Y_test_pred_proba = self.model.predict_proba(self.X_test)
        print(
            "Model evaluation on test set completed, check model attributes for results"
        )

    def evaluate_documnet(self, document: str):
        """Predicts the rating for an input string given a trained model"""
        document_embeddings = self.vectorizer.transform(
            [document]
        )  # vectorizer expects a list of strings
        return self.model.predict_proba(document_embeddings), self.model.predict(
            document_embeddings
        )

    def export_model(self):
        """Exports the custom model's sklearn linear model"""
        output_config = self.config.output.output_path
        ModelSaving().save_model_with_timestamp(
            self.vectorizer, self.model, output_config
        )
