from collections import namedtuple as nt
import math

class StatsKeeper :
    _BODYWEIGHT = 0.3

    class Document :
        @property
        def name(self) :
            return self._name
        
        @property
        def tfs(self) :
            return self._tfs
        
        @property
        def tf_idfs(self) :
            return self._tfidfs

        def __init__(self, title, tfs, tfidfs) :
            self._name = title
            self._tfs = tfs
            self._tfidfs = tfidfs

    def load_document(self, title, processed_title, processed_text) :
        tfs = self._tfs(processed_title, processed_text)

        for token in tfs.keys() :
            self._wdf[token] = self._wdf.get(token, 0) + 1
        self._sortedwdf = sorted(self._wdf)

        idfs = self._idfs()
        tfidfs = self._tfidfs(idfs, tfs)
        self._documents[title] = self.Document(title, tfs, tfidfs)

        for title, document in self._documents.items() :
            document._tfidfs = self._tfidfs(idfs, document.tfs)


    def _tfs(self, processed_title, processed_text) :
        term_frequency = {}
        term_appearance = {}
        for i, text in enumerate([processed_title, processed_text]) :
            for token in text :
                term_frequency[token] = term_frequency.get(token, 0) + 1
                if i == 0 and token not in term_appearance:
                    term_appearance[token] = 1
                elif i == 1 and token not in term_appearance:
                    term_appearance[token] = 0
                elif i == 1 and term_appearance[token] == 1 :
                    term_appearance[token] = 2

        
        for term in term_frequency.keys() :
            if term_appearance[token] == 1 :
                alpha = 1 - self._BODYWEIGHT
            elif term_appearance[token] == 2 :
                alpha = 1
            elif term_appearance[token] == 0 :
                alpha = self._BODYWEIGHT

            term_frequency[term] = alpha *term_frequency[term] /(len(processed_title)+len(processed_text))
        return term_frequency

    def _idfs(self) :
        idfs  = {}
        for key, value in self._wdf.items() :         
            idfs[key] = math.log((len(self._documents)+1)/(1+value))

        return idfs

    def _tfidfs(self,idfs, tfs) :
        tfidfs = {}
        for key in tfs.keys() :
            tfidfs[key] = idfs[key]*tfs[key]
        
        return tfidfs

    def query_matching_score(self, text_of_query) :
        scores = {}
        for title, document in self._documents.items() :
            for token in text_of_query :
                if token in document.tf_idfs :
                    scores[title] = scores.get(title, 0)+ document.tf_idfs[token]
        results = sorted(scores.items(), key=lambda x : x[1], reverse=True)
        return results


    def __init__(self) :
        self._documents = {}
        self._wdf = {}