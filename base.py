from dataset_loader import Dataset
from document_preprocessor import DocumentPreprocessor
from document_vectors import StatsKeeper

'''
Little simple showcase app showing the differences in searches
using "matching score" and "cosine similarity"
'''

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

while 1 :
    print("\nType \"__exit__\" if you want to leave.")
    query = input("What are you searching for ? : ")
    if query == "__exit__":
        break
    preprocessed_query = documentPreprocessor.preprocess_document(text=query, title="query", path="", save_cached=False)
    results = statsKeeper.query_matching_score(preprocessed_query.text)
    print("BEST MATCHES ACCORDING TO MATCHING SCORE: ")
    for rank, (title, score) in enumerate(results[:5]) :
        print("{}. {} : {}".format(rank+1, title, score))

    results = statsKeeper.query_cosine_similarity(preprocessed_query.text)
    print("BEST MATCHES ACCORDING TO COSINE SIMILARITY: ")
    for rank, (title, score) in enumerate(results[:5]) :
        print("{}. {} : {}".format(rank+1, title, score))
