import math
import numpy as np
from enum import Enum


class StatsKeeper:

    class Document:
        @property
        def name(self):
            return self._name

        @property
        def term_frequencies(self):
            return self._tfs

        @property
        def sorted_term_frequencies(self):
            return self._sorted_tfs

        @property
        def sorted_tf_idfs(self):
            return self._sorted_tf_idfs

        @property
        def tf_idfs(self):
            return self._tf_idfs

        @property
        def tf_idfs_vector(self):
            return self._vector

        @tf_idfs.setter
        def tf_idfs(self, value):
            if self._statsKeeper.compiled:
                raise Exception("Cannot change tf_idfs of document in an already compiled statsKeeper !")
            self._tf_idfs = value
            self._sorted_tf_idfs = sorted(value.items(), key=lambda x: x[1], reverse=True)
        
        def _initialize_vector(self, vocab, length):
            self._vector = np.zeros((length,), dtype=float)
            for word, value in self._tf_idfs.items():
                index = vocab[word]
                self._vector[index] = value

        def __init__(self, title, tfs, stats_keeper):
            self._name = title
            self._tfs = tfs
            self._sorted_tfs = sorted(tfs.items(), key=lambda x: x[1], reverse=True)
            self._tf_idfs = {}
            self._sorted_tf_idfs = {}
            self._vector = []
            self._statsKeeper = stats_keeper

    @property
    def compiled(self):
        return self._compiled

    @property
    def documents(self):
        return self._documents

    def __init__(self, bodyweight: float = 0.3):
        self._bodyweight = bodyweight
        self._documents = {}
        self._wdf = {}
        self._sorted_wdf = {}
        self._sorted_idfs = {}
        self._vocab = {}
        self._len_vocab = 0
        self._compiled = False

    # during load of each document the following stats are
    # collected : term frequencies of each token
    # the following stats are incremented :
    # document frequencies of each token in the document
    def load_document(self, title="", processed_title="", processed_text=""):
        if self._compiled:
            raise Exception("Cannot load new documents to already compiled stats keeper!")

        tfs = self._calculate_document_tfs(processed_title, processed_text)
        for token in tfs.keys():
            self._wdf[token] = self._wdf.get(token, 0) + 1
        self._documents[title] = self.Document(title, tfs, self)

    def compile(self):
        for index, (key, _) in enumerate(self._wdf.items()):
            self._vocab[key] = index
        self._len_vocab = len(self._vocab)

        idfs = self._calculate_idfs()
        for document in self._documents.values():
            document.tf_idfs = StatsKeeper._calculate_tfidfs(
                                        idfs,
                                        document.term_frequencies)
            document._initialize_vector(self._vocab, self._len_vocab)

        self._sorted_wdf = sorted(self._wdf.items(), key=lambda x: x[1], reverse=True)
        self._sorted_idfs = sorted(idfs.items(), key=lambda x: x[1], reverse=True)
        self._compiled = True

    def _calculate_document_tfs(self, processed_title, processed_text):
        term_frequency = {}
        appearance = Enum('appearance', 'both title body')
        term_appearance = {}

        # using WEIGHTED term frequency, there is need to
        # determine if a token was found in title/body/both
        for token in processed_title:
            term_frequency[token] = term_frequency.get(token, 0) + 1
            term_appearance[token] = appearance.title

        for token in processed_text:
            term_frequency[token] = term_frequency.get(token, 0) + 1
            if token not in term_appearance:
                term_appearance[token] = appearance.body
            else:
                term_appearance[token] = appearance.both

        # static class argument _BODYWEIGHT determines how much more
        # important is a token in title than a token outside
        for token in term_frequency.keys():
            alpha = 0
            if term_appearance[token] == appearance.title:
                alpha = 1 - self._bodyweight
            elif term_appearance[token] == appearance.both:
                alpha = 1
            elif term_appearance[token] == appearance.body:
                alpha = self._bodyweight

            term_frequency[token] = alpha * term_frequency[token] / (len(processed_title) + len(processed_text))
        return term_frequency

    def _calculate_idfs(self):
        idfs = {}
        for key, value in self._wdf.items():
            # use normalized version of idf (divided by 1+value to avoid dividing by zero)
            idfs[key] = math.log(len(self._documents) / (1 + value)) + 1

        return idfs

    @staticmethod
    def _calculate_tfidfs(idfs, tfs):
        tfidfs = {}
        for key in tfs.keys():
            tfidfs[key] = idfs[key] * tfs[key]

        return tfidfs

    def query_matching_score(self, text_of_query):
        scores = {}

        # we create score for each document in the corpus,
        # the score is simply the sum of all the tf-idfs of
        # tokens in the document from the query
        for title, document in self._documents.items():
            for token in text_of_query:
                if token in document.tf_idfs:
                    scores[title] = scores.get(title, 0) + document.tf_idfs[token]

        results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return results

    def query_cosine_similarity(self, text_of_query):
        scores = {}

        # we create vector of the query
        query_vector = np.zeros((self._len_vocab,), dtype=float)

        token_freq = {}
        for token in text_of_query:
            token_freq[token] = token_freq.get(token, 0) + 1

        for token, freq in token_freq.items():
            document_freq = self._wdf.get(token, 0)
            idf = math.log(len(self._documents) / (1+document_freq)) + 1

            # if the searched token isn't in the vocab it won't be included
            # in the searched query
            try:
                index = self._vocab[token]
                query_vector[index] = freq * idf
            except IndexError:
                pass

        # next we perform cosine similarity calculation of the query with all the documents
        for title, document in self._documents.items():
            scores[title] = np.dot(query_vector, document.tf_idfs_vector) / (
                    np.linalg.norm(query_vector)*np.linalg.norm(document.tf_idfs_vector))

        results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return results
