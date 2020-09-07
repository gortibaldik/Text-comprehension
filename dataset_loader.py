# loader of custom data from dataset : http://archives.textfiles.com/stories.zip

# following tutorial from
# https://towardsdatascience.com/tf-idf-for-document-ranking-from-scratch-in-python-on-real-world-dataset-796d339a4089


import os
import io
import re
import urllib.request as url_request
import zipfile
import sys
from collections import namedtuple


class Dataset:
    _URL = "http://archives.textfiles.com/stories.zip"
    _INDEX_NAME = "index.html"
    _NAMED_TUPLE = namedtuple("document", ["title", "text"])

    @property
    def texts(self):
        return self._text_dictionary

    @property
    def show_errors(self):
        return self._show_errors

    @show_errors.setter
    def show_errors(self, value: bool):
        self._show_errors = value

    def __init__(self):
        # downloading dataset and unzipping it
        # removing prefix and extension
        dir_name = os.path.splitext(os.path.basename(self._URL))[0]
        if not os.path.isdir(dir_name):
            print("Downloading dataset {}...".format(dir_name), file=sys.stderr)
            with url_request.urlopen(self._URL) as requested:
                with zipfile.ZipFile(io.BytesIO(requested.read())) as zip_file:
                    zip_file.extractall()

        # the directory stories is included in subdirs of stories
        subdirs = ["stories"] + [os.path.join(dir_name, d) for subdir in os.walk(dir_name)
                                 for d in subdir[1]]
        self._show_errors = False
        self._text_dictionary = {}
        correct = incorrect =0
        for subdir in subdirs:
            index_name = os.path.join(subdir, self._INDEX_NAME)
            if not os.path.exists(index_name):
                raise Exception("{} not present in {} -- directory corrupted !"
                                .format(self._INDEX_NAME, subdir))

            # extract the names of the documents and their titles
            with open(index_name, 'r') as index_file :
                text = index_file.read()
                file_names = re.findall('><A HREF="(.*)">[^<]*</A> ', text)
                file_titles = re.findall('<BR><TD> (.*)\n', text)

            if len(file_names) != len(file_titles) :
                raise Exception("len(file_names) != len(file_titles) -- directory corrupted !")

            # extract texts from specified paths
            for file_name, title in zip(file_names, file_titles):
                path = os.path.join(subdir, file_name)
                try :
                    with open(path, 'r') as file:
                        to_insert = file.read().strip()
                        self._text_dictionary[path] = Dataset._NAMED_TUPLE._make([title, to_insert])
                    correct += 1
                except:
                    if self.show_errors:
                        print("ERROR IN DECODING {}".format(path))
                    incorrect += 1

        print("Statistics of loading : correct : {}\tincorrect : {}".format(correct, incorrect))
