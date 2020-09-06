import math
import os   

class StatsKeeperBasic:
    class Document :
        def __init__(self, stats,text, path) :
            for _,(stat,value) in enumerate(stats.items()) :
                setattr(self, stat, value)
            self._name = os.path.basename(path)
            self._text = text
            self._stats = stats

        @property 
        def stats(self) :
            return self._stats

        @property
        def text(self) :
            return self._text
        
        @property
        def name(self) :
            return self._name

        def term_frequency(self, word) :
            return self._text.words.count(word)

        def normalized_term_frequency(self, word) :
            return self._text.words.count(word) / len(self._text.words)

        def get_nhighest_tfidf(self,n) :
            return enumerate(self.tf_idfs[:n])



    def _all_tfs(self, document) :
        tfs = {}
        for word in document.words :
            tfs[word] = tfs.get(word, 0) + 1
        return tfs

    def load_document(self, path) :
        with open(path, "r", encoding="utf-8") as in_file :
            document = TextBlob(in_file.read())
        stats = {}
        stats["term_frequencies"] = sorted(self._all_tfs(document).items(),key=lambda x:x[1], reverse=True) 
        stats["normalized_tfs"] = sorted({word : frequency / len(document.words) for word, frequency in stats["term_frequencies"]}.items(), key=lambda x:x[1], reverse=True)
        stats["tf_idfs"] = []
        self._documents[path] = self.Document(stats, document, path)
        
        for word, _ in stats["term_frequencies"] :
            self._wordsDF[word] = self._wordsDF.get(word, 0) + 1

        for word, _ in self._wordsDF.items() :
            self._wordsIDF[word] = math.log(len(self._documents)/(1+self._wordsDF[word]))
        
        for doc in self._documents.values() :
            tmp = {}
            for word, value in doc.normalized_tfs :
                tmp[word] = value * self._wordsIDF[word]
            doc.tf_idfs = sorted(tmp.items(), key=lambda x:x[1], reverse=True)
            
    @property
    def documents(self) :
        return self._documents

    @property
    def wordDFs(self) :
        return self._wordsDF

    @property
    def wordIDFs(self) :
        return self._wordsIDF

    def __init__(self) :
        self._documents = {}
        self._wordsDF = {}
        self._wordsIDF = {}