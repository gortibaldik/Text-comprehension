# Keyword extraction using TF-IDF

- module statsKeeperBasic.py
Basic module providing tools for key-word extraction. Doesn't use any NLP tools only statistics. No stemming, no lemmatization.
- class ```StatsKeeperBasic```
	inner class ```Document``` 
		properties :
		- ```text```
		- ```name```
		- ```term_frequencies``` - sorted list of number of occurencies of each distinct word
		- ```normalized_tfs``` - same as term_frequencies, each term divided by number of all words
		- ```tf_idfs``` - normalized_tfs * inverse_document_frequencies 
			calculated on the fly by StatsKeeperBasic
	properties :
	- ```documents```
	- ```wordDFs``` - as documents are loaded into StatsKeeperBasic we keep track of all distinct words in
		documents and counts of number of documents they appeared in
	- ```wordIDFs``` - inverse document frequencies calculated on the fly similary to ```wordDFs```

	methods :
	- ```load_document```- loads document and recalculates all the necessary statistics  
