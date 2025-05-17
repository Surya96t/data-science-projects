import datetime
import os
import pickle


class ModelSaving(object):
    @staticmethod
    def get_current_timestamp():
        return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    @staticmethod
    def save_model_with_timestamp(vectorizer, model, output_config):
        filename = ModelSaving.get_current_timestamp() + "_LogReg" + ".pickle"
        filepath = os.path.join(output_config, filename)
        with open(filepath, "wb") as outputfiile:
            pickle.dump((vectorizer, model), outputfiile)

        return print(f"Saved vectorizer and model to pickle file at {filepath}")


if __name__ == "__main__":
    print(ModelSaving.get_current_timestamp())
