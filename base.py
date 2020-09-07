from dataset_loader import Dataset
from document_preprocessor import DocumentPreprocessor
from document_vectors import StatsKeeper
import random
# from num2words import num2words

dataset = Dataset()
print("DATASET LOADED !")

documentPreprocessor = DocumentPreprocessor(remove_apostrophes=True,
                                            remove_punctuation=True,
                                            remove_single_characters=True,
                                            remove_stop_words=True,
                                            stemming=True,
                                            number_converting=True,
                                            lower_case=True)
statsKeeper = StatsKeeper()

for path, (title, text) in dataset.texts.items():
    preprocessed = documentPreprocessor.preprocess_document(path=path, text=text, title=title)
    statsKeeper.load_document(title, preprocessed.title, preprocessed.text)
print("DATASET PREPARED FOR COMPILATION !")
statsKeeper.compile()
print("DATASET COMPILED !")

print("Word document frequencies:")
for rank, (token, frequency) in enumerate(statsKeeper._sorted_wdf[:10]):
    print("{}. {} : {}".format(rank + 1, token, frequency))

print("Inverse document frequencies:")
for rank, (token, frequency) in enumerate(statsKeeper._sorted_idfs[:10]):
    print("{}. {} : {}".format(rank + 1, token, frequency))

print("Now we will pick random document from the corpus")
_, (title, text) = random.choice(list(dataset.texts.items()))
print("TITLE : {}".format(title))
print("\n\n\nTEXT : {}".format(text))

print("Most frequent tokens: ")
for rank, (token, frequency) in enumerate(statsKeeper.documents[title].sorted_term_frequencies[:10]):
    print("{}. {} : {}".format(rank + 1, token, frequency))

print("Most valuable tokens:")
for rank, (token, frequency) in enumerate(statsKeeper.documents[title].sorted_tf_idfs[:10]):
    print("{}. {} : {}".format(rank + 1, token, frequency))
# while 1 :
#     query = input("What are you searching for ? : ")
#     preprocessed_query = documentPreprocessor.preprocess_document(text=query, title="query")
#     results = statsKeeper.query_matching_score(preprocessed_query.p_text)
#     print("BEST MATCHES : ")
#     for rank, (title, score) in enumerate(results[:5]) :
#         print("{}. {} : {}".format(rank+1, title, score))
