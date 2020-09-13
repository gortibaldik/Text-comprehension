from text_classifiers.dataset_loader import Dataset
from document_vectors import StatsKeeper
from nltk.tokenize import word_tokenize

Dataset._ORIGINAL_CSV = "text_classifiers/Reviews.csv"
dataset = Dataset()
statsKeeper = StatsKeeper()

print("Number of data: {}".format(len(dataset._data['cleaned_text'])))
for index, text in enumerate(dataset._data['cleaned_text']):
    text = word_tokenize(text)
    statsKeeper.load_document(title=str(index), processed_text=text)

statsKeeper.compile()

