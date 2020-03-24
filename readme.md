# Keyword extraction using TF-IDF

- module ```base.py```
	- the main module for testing functionality of tf-idf on dataset
- module ```dataset_loader.py```
	- not finalized yet, at mean time it'll implement download and processing of dataset from http://archives.textfiles.com/stories.zip
	right now you need to specify location of stories.zip file on your disc in constructor of ```Dataset``` class
- module ```document_preprocessor.py```
	- main module for text preprocessing
	- with use of libraries : ```nltk```, ```num2words```, ```numpy``` performs basic text preprocessing steps such as 
		- lowercasing
		- removing stop words
		- stemming
- module ```document_vectors.py```
	- loads preprocessed text and creates tf_idf stats for each document loaded
	- inner class ```Document``` holding stats for each document separately
- module ```statsKeeperBasic.py```
- Basic module providing tools for key-word extraction. Doesn't use any NLP tools only statistics. No stemming, no lemmatization.
- class ```StatsKeeperBasic```
	- inner class ```Document``` 
		-properties :
		- ```text```
		- ```name```
		- ```term_frequencies``` - sorted list of number of occurencies of each distinct word
		- ```normalized_tfs``` - same as term_frequencies, each term divided by number of all words
		- ```tf_idfs``` - normalized_tfs * inverse_document_frequencies 
			calculated on the fly by StatsKeeperBasic
	- properties :
	- ```documents```
	- ```wordDFs``` - as documents are loaded into StatsKeeperBasic we keep track of all distinct words in
		documents and counts of number of documents they appeared in
	- ```wordIDFs``` - inverse document frequencies calculated on the fly similary to ```wordDFs```

	methods :
	- ```load_document```- loads document and recalculates all the necessary statistics  
