import numpy
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from num2words import num2words
from collections import namedtuple as nt


class DocumentPreprocessor:
    _STOP_WORDS = stopwords.words('english')
    _NAMED_TUPLE = nt("document", ["title", "text"])

    # initialize all the flags used for document preprocessing
    def __init__(self,
                 lower_case=False,
                 remove_stop_words=False,
                 remove_punctuation=False,
                 remove_apostrophes=False,
                 remove_single_characters=False,
                 stemming=False,
                 lemmatization=False,
                 number_converting=False):
        self._lc = lower_case
        self._sw = remove_stop_words
        self._p = remove_punctuation
        self._a = remove_apostrophes
        self._sc = remove_single_characters
        self._st = stemming
        self._lem = lemmatization
        self._nc = number_converting
        self._stemmer = PorterStemmer()
        self._file_suffix = self._create_parameters_suffix()

    # traverse through document and apply all the
    # preprocessing steps specified in the constructor
    # for faster working it may save preprocessed documents
    # to disc (to the same directory as the original dataset)
    def preprocess_document(self, path, text, title):
        # the adequately preprocessed document doesn't exist yet
        if not os.path.exists(self._create_parameters_filename(path)):
            arr = [title, text]
        # document has already been processed
        else:
            arr = [title]

        for n, item in enumerate(arr):
            if self._lc:  # make the text lowercase
                item = DocumentPreprocessor._lower(item)
            if self._p:  # remove punctuation
                item = DocumentPreprocessor._remove_punctuation(item)

            def sub_preprocess(_item):
                tokenized = word_tokenize(_item)
                _item = ""
                for token in tokenized:
                    if self._sw:  # remove stop words
                        token = DocumentPreprocessor._erase_stopword(token)
                    if self._sc:  # remove short words
                        token = self._erase_short_length(token)
                    if self._nc:  # rename numbers
                        token = self._try_convert_number(token)
                    if len(token) > 0:
                        _item += " " + token

                return str(_item)

            item = sub_preprocess(item)
            if self._a:
                item = DocumentPreprocessor._remove_apostrophe(item)

            if self._st:  # stem the verbs and conjugations
                item = self._stem(item)
                item = sub_preprocess(item)

            arr[n] = item

        # save for future use
        if len(arr) == 2:
            preprocessed_text = arr[1]
            with open(self._create_parameters_filename(path), 'w') as file:
                for w in preprocessed_text: file.write(w)
        # if the text is already saved, load it from disc
        else:
            with open(self._create_parameters_filename(path), 'r') as file:
                preprocessed_text = file.read()
        preprocessed_title = arr[0]
        return DocumentPreprocessor._NAMED_TUPLE._make([word_tokenize(preprocessed_title), word_tokenize(preprocessed_text)])

    # stem all the words in the text
    def _stem(self, text):
        tokenized_text = word_tokenize(text)
        stemmed = ""
        for token in tokenized_text:
            stemmed += " " + self._stemmer.stem(token)

        stemmed = self._remove_punctuation(stemmed)
        return str(stemmed)

    def _create_parameters_filename(self, path):
        wo_ext, ext = os.path.splitext(path)
        file_name = wo_ext + self._file_suffix
        return file_name + ext

    def _create_parameters_suffix(self):
        suffix = "_"
        suffix += "lc" if self._lc else ""
        suffix += "sw" if self._sw else ""
        suffix += "p" if self._p else ""
        suffix += "a" if self._a else ""
        suffix += "sc" if self._sc else ""
        suffix += "st" if self._st else ""
        suffix += "lem" if self._lem else ""
        suffix += "nc" if self._nc else ""
        return suffix

    @staticmethod
    def _lower(text):
        return str(numpy.char.lower(text))

    @staticmethod
    def _remove_punctuation(text):
        symbols = "!\"#$%&()*+-./:;<=>?@[\\]^_`{|}~\n"
        return str(text.translate(str.maketrans('', '', symbols)))

    @staticmethod
    def _remove_apostrophe(text):
        return str(numpy.char.replace(text, "'", " "))

    @staticmethod
    def _erase_stopword(word):
        if word in DocumentPreprocessor._STOP_WORDS:
            word = ""
        return word

    @staticmethod
    def _erase_short_length(word):
        if len(word) <= 1:
            word = ""
        return word

    @staticmethod
    def _try_convert_number(word):
        try:
            return num2words(int(word))
        except ValueError:  # catching only errors from int(word)
            return word
