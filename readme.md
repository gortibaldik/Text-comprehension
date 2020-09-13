# Summarization of the text using ```RNN```
- comparing of several methods and several neural network architectures on the [Amazon Fine Foods Review dataset](https://www.kaggle.com/snap/amazon-fine-food-reviews)
- the results are summarized in ```Main.ipynb```
- the code needed is in the package ```text_classifiers```
- using [pytextrank](https://github.com/DerwenAI/pytextrank)

# Keyword extraction using TF-IDF
- this little library shows some results using TF-IDF on a real word [dataset](http://archives.textfiles.com/stories.zip
)
- the results are summarized in ```TF-IDFNotebook.ipynb```

- module ```base.py```
	- shows how ```TF-IDF``` can be used for querying the dataset
	- utilizes 2 methods : ```matching score``` and ```cosine similarity```

- module ```dataset_loader.py```
	- downloads [the real word dataset](http://archives.textfiles.com/stories.zip)
	- unzips the file and creates new directories storing all the documents

- module ```document_preprocessor.py```
	- main module for text preprocessing
	- with use of libraries : ```nltk```, ```num2words```, ```numpy``` performs basic text preprocessing steps such as 
		- lowercasing
		- removing punctuation
		- removing apostrophes
		- removing single-letter words
		- removing stop words
		- stemming
		- normalizing numbers to one uniform format
	- it also gives user a possibility of saving all the changes on the disc, which may save 
	time during next preprocessing

- module ```document_vectors.py``` with class ```StatsKeeper```
	- loads preprocessed text and incrementally holds the ```word document frequency``` for 
each word in the corpus
	- class ```Document``` holding stats for each document separately, namely
		- ```tf-idf```
		- ```tf-idf vectorised```
		- ```tf```
	- since the class holds all the documents in human readable and also in vectorised form, it is
possible to use this class for vector-like operations such as computing ```cosine-similarity```
