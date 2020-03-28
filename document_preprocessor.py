import numpy
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from num2words import num2words
from collections import namedtuple as nt

# in constructor we choose which preprocessing will be applied to loaded documents
class DocumentPreprocessor :

    # probably I'll change the defaults to True
    def __init__(self, 
            lower_case=False, 
            remove_stop_words=False, 
            remove_punctuation=False, 
            remove_apostrophes=False, 
            remove_single_characters=False, 
            stemming = False, 
            lemmatization=False, 
            number_converting=False) :
        self._lc = lower_case
        self._sw = remove_stop_words
        self._p = remove_punctuation
        self._a = remove_apostrophes
        self._sc = remove_single_characters
        self._st = stemming
        self._lem = lemmatization
        self._nc = number_converting
        self._stop_words = stopwords.words('english')
        self._stemmer = PorterStemmer()
    
    def preprocess_document(self, text, title) :
        if self._lc : 
            text = self._lower(text)
            title = self._lower(title)
        if self._p :
            text = self._remove_punctuation(text)
            title = self._remove_punctuation(title)

        preprocessed_text = self._preprocess(text)
        preprocessed_title = self._preprocess(title)

        if self._st :
            preprocessed_text = self._stem(preprocessed_text)
            preprocessed_title = self._stem(preprocessed_title)

        type_of_tuple = nt("document", ["title", "p_title", "p_text"])
        return type_of_tuple._make([title, word_tokenize(preprocessed_title), word_tokenize(preprocessed_text)])
        



    @property
    def documents(self) :
        return self._documents

    def _stem(self, text) :
        tokenized_text = word_tokenize(text)
        stemmed = ""
        for token in tokenized_text :
            stemmed += " " + self._stemmer.stem(token)

        stemmed = self._remove_punctuation(stemmed)
        stemmed = self._preprocess(stemmed)
        return str(stemmed)

    def _preprocess(self, text) :
        tokenized_text = word_tokenize(text)
        preprocessed = ""
        for token in tokenized_text :
            if self._sw :
                token = self._check_stopWords(token)
            if self._sc :
                token = self._check_length(token)
            if self._nc :
                token = self._check_numbers(token)
            if len(token) > 0 :
                preprocessed += " " + token

        if self._a :
            preprocessed = self._remove_apostrophe(preprocessed)
        return str(preprocessed)

    def _lower(self, text) :
        return str(numpy.char.lower(text))

    def _remove_punctuation(self, text) :
        symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
        return str(text.translate(str.maketrans('','',symbols)))

    def _remove_apostrophe(self, text) :
        return str(numpy.char.replace(text, "'", " "))

    def _check_stopWords(self, word) :
        if word in self._stop_words :
            word = ""
        return word
    
    def _check_length(self, word) :
        if len(word) <= 1 :
            word = ""
        return word

    def _check_numbers(self, word) :
        try :
            return num2words(int(word))
        except :
            return word