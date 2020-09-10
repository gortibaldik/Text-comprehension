import pandas as pd # used for loading .csv file
import numpy as np
import matplotlib.pyplot as plt
import os
import json

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from dataset_cleaning import contraction_mapping, text_cleaner
from sklearn.model_selection import train_test_split

class Dataset:
    _PREPARED_CSV = "Reviews_prepared.csv"
    _PREPARED_TOKEN_DATA = "token_data_prepared.json"
    _PREPARED_TOKEN_LABEL = "token_label_prepared.json"
    _PREPARED_TRAIN_DATA = "dataset_train_data_prepared.npy"
    _PREPARED_TRAIN_LABEL = "dataset_train_label_prepared.npy"
    _PREPARED_TEST_DATA = "dataset_test_data_prepared.npy"
    _PREPARED_TEST_LABEL = "dataset_test_label_prepared.npy"

    def _load(self):
        if os.path.exists(Dataset._PREPARED_CSV):
            self._data = pd.read_csv(Dataset._PREPARED_CSV)
            return False
        self._data = pd.read_csv("Reviews.csv", nrows=1000)
        self._data.drop_duplicates(subset=['Text'], inplace=True)
        self._data.dropna(axis=0, inplace=True)  # drop all the rows containing NA values
        return True

    def _prepare(self):
        cleaned_text = []
        for txt in self._data['Text']:
            cleaned_text.append(text_cleaner(txt))

        cleaned_summary = []
        for txt in self._data['Summary']:
            cleaned_summary.append("_START_ "+text_cleaner(txt)+" _END_")

        self._data['cleaned_text'] = cleaned_text
        self._data['cleaned_summary'] = cleaned_summary
        self._data['cleaned_summary'].replace("_START_  _END_", np.nan, inplace=True)
        self._data.dropna(axis=0, inplace=True)
        self._data.to_csv(Dataset._PREPARED_CSV)

    def _create_dataset(self):
        datapoint_train, datapoint_test, label_train, label_test = \
            train_test_split(self._data['cleaned_text'], self._data['cleaned_summary'],
                             test_size=0.1, random_state=0, shuffle=True)

        datapoint_tokenizer = Tokenizer()
        label_tokenizer = Tokenizer()

        if os.path.exists(Dataset._PREPARED_TOKEN_DATA):
            with open(Dataset._PREPARED_TOKEN_DATA, 'r') as fp:
                datapoint_tokenizer.word_index = json.load(fp)
            datapoint_tokenizer.index_word = dict([(i, char) for char, i in
                                                             datapoint_tokenizer.word_index.items()])
        else:
            datapoint_tokenizer.fit_on_texts(list(datapoint_train))
            with open(Dataset._PREPARED_TOKEN_DATA, 'w') as fp:
                json.dump(datapoint_tokenizer.word_index, fp)

        if os.path.exists(Dataset._PREPARED_TOKEN_LABEL):
            with open(Dataset._PREPARED_TOKEN_LABEL, 'r') as fp:
                label_tokenizer.word_index = json.load(fp)
            label_tokenizer.index_word = dict([(i, char) for char, i in
                                                         label_tokenizer.word_index.items()])
            
        else:
            label_tokenizer.fit_on_texts(list(label_train))
            with open(Dataset._PREPARED_TOKEN_LABEL, 'w') as fp:
                json.dump(label_tokenizer.word_index, fp)


        self.max_len_datapoint = 80
        
        if os.path.exists(Dataset._PREPARED_TRAIN_DATA):
            self.datapoint_train = np.load(Dataset._PREPARED_TRAIN_DATA)
        else:
            self.datapoint_train = pad_sequences(datapoint_tokenizer.texts_to_sequences(datapoint_train),
                                                 maxlen=self.max_len_datapoint, padding='post')
            np.save(Dataset._PREPARED_TRAIN_DATA, self.datapoint_train)
        
        if os.path.exists(Dataset._PREPARED_TEST_DATA):
            self.datapoint_test = np.load(Dataset._PREPARED_TEST_DATA)
        else:
            self.datapoint_test  = pad_sequences(datapoint_tokenizer.texts_to_sequences(datapoint_test),
                                                 maxlen=self.max_len_datapoint, padding='post')
            np.save(Dataset._PREPARED_TEST_DATA, self.datapoint_test)


        self.datapoint_vocab_size = len(datapoint_tokenizer.word_index) + 1
        self.datapoint_tokenizer = datapoint_tokenizer
        self.label_tokenizer = label_tokenizer

        self.max_len_label = 10
        if os.path.exists(Dataset._PREPARED_TRAIN_LABEL):
            self.label_train = np.load(Dataset._PREPARED_TRAIN_LABEL)
        else:
            self.label_train = pad_sequences(label_tokenizer.texts_to_sequences(label_train),
                                             maxlen=self.max_len_label, padding='post')
            np.save(Dataset._PREPARED_TRAIN_LABEL, self.label_train)

        if os.path.exists(Dataset._PREPARED_TEST_LABEL):
            self.label_test = np.load(Dataset._PREPARED_TEST_LABEL)
        else:
            self.label_test  = pad_sequences(label_tokenizer.texts_to_sequences(label_test),
                                             maxlen=self.max_len_label, padding='post')
            np.save(Dataset._PREPARED_TEST_LABEL, self.label_test)

        self.label_vocab_size = len(label_tokenizer.word_index) + 1

    def __init__(self):
        if self._load():
            self._prepare()
        self._create_dataset()
