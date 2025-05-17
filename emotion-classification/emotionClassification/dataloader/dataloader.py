import pandas as pd
import numpy as np
import re

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer


# nltk.download("punkt")
# nltk.download("wordnet")


class DataLoader:
    """DataLoader class to load the data and preprocess it"""

    @staticmethod
    def load_data(data_config):
        """Loads the data from the given path"""
        df = pd.read_csv(data_config.path)
        return df

    @staticmethod
    def lowercase_text(data_config, dataset=None):  # dataset -> df
        """Converts the text to lowercase"""
        dataset["lowercase_text"] = dataset[data_config.x].str.lower()
        return dataset

    @staticmethod
    def remove_special_characters(data_config, dataset=None):  # dataset -> df
        """Removes special characters from the text"""

        def remove_patterns(text):
            # Remove URLs
            text = re.sub(r"http[s]?://\S+", "", text)
            # Remove markdown-style links
            text = re.sub(r"\[.*?\]\(.*?\)", "", text)
            # Remove handles (that start with '@')
            text = re.sub(r"@\w+", "", text)
            # Remove punctuation and other special characters
            text = re.sub(r"[^\w\s]", "", text)
            return text.strip()

        dataset["lowercase_text"] = dataset["lowercase_text"].apply(remove_patterns)

        return dataset

    @staticmethod
    def tokenize_text(data_config, dataset=None):
        """Tokenizes the text"""
        dataset["tokenized_text"] = dataset["lowercase_text"].apply(word_tokenize)

        return dataset

    @staticmethod
    def lemmatize_text(data_config, dataset=None):
        """Lemmatizes the text"""

        def get_wordnet_pos(word):
            """Map POS tag to first character lemmatize() accepts"""
            if word.startswith("J"):
                return wordnet.ADJ
            elif word.startswith("V"):
                return wordnet.VERB
            elif word.startswith("N"):
                return wordnet.NOUN
            elif word.startswith("R"):
                return wordnet.ADV
            else:
                return wordnet.NOUN  # default to noun

        lemmatizer = WordNetLemmatizer()
        dataset["lemmatized_text"] = dataset["tokenized_text"].apply(
            lambda x: [
                lemmatizer.lemmatize(w, get_wordnet_pos(t)) for w, t in nltk.pos_tag(x)
            ]
        )

        # join back the tokens into a string
        dataset["lemmatized_text"] = dataset["lemmatized_text"].apply(
            lambda x: " ".join(x)
        )

        return dataset

    @staticmethod
    def preprocess_data(data_config, dataset=None):
        """Preprocesses the data and splits it into train and test sets"""
        dataset = DataLoader.lowercase_text(data_config, dataset)
        dataset = DataLoader.remove_special_characters(data_config, dataset)
        dataset = DataLoader.tokenize_text(data_config, dataset)
        dataset = DataLoader.lemmatize_text(data_config, dataset)

        x = dataset["lemmatized_text"]
        y = dataset[data_config.y]

        # mapping the labels to integers
        y = y.map({"anger": 0, "joy": 1})

        test_size = data_config.test_size
        random_state = data_config.random_state

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=random_state
        )

        # vectorize the text
        ngram_list = data_config.ngram_range
        ngram_tuple = tuple(ngram_list)
        # print("This is the ngram-range: ", ngram_tuple)
        vectorizer = TfidfVectorizer(ngram_range=ngram_tuple)

        # fit_transform on training texts
        X_train = vectorizer.fit_transform(x_train)
        X_test = vectorizer.transform(x_test)
        Y_train = np.array(y_train)
        Y_test = np.array(y_test)

        return (
            x,
            y,
            x_train,
            x_test,
            y_train,
            y_test,
            X_train,
            X_test,
            Y_train,
            Y_test,
            vectorizer,
        )
