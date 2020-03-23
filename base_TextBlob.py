import argparse
from statsKeeperBasic import StatsKeeperBasic

parser = argparse.ArgumentParser()
parser.add_argument('--documents',nargs='+')
args = parser.parse_args([] if "__file__" not in globals() else None)
statskeeper = StatsKeeperBasic()

for path in args.documents :
    statskeeper.load_document(path)

for path, document in statskeeper.documents.items() :
    print("Top words in document : {}".format(document.name))
    for i, (word, w_tfidf) in document.get_nhighest_tfidf(3) :
        print("Word: {}, \t{}TF-IDF: {}".format(word,"\t" if len(word) < 8 else "",  w_tfidf))

