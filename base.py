from dataset_loader import Dataset
from document_preprocessor import DocumentPreprocessor
from document_vectors import StatsKeeper
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
    print(title)
    print(preprocessed.title)
    print("----")
    print(text)
    print("-\n@\n-\n")
    print(preprocessed.text)
    input("== == == ==")
    statsKeeper.load_document(preprocessed.title, preprocessed.title, preprocessed.text)
print("DATASET VECTORIZED !")

# while 1 :
#     query = input("What are you searching for ? : ")
#     preprocessed_query = documentPreprocessor.preprocess_document(text=query, title="query")
#     results = statsKeeper.query_matching_score(preprocessed_query.p_text)
#     print("BEST MATCHES : ")
#     for rank, (title, score) in enumerate(results[:5]) :
#         print("{}. {} : {}".format(rank+1, title, score))
