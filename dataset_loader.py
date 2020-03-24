# loader of custom data from dataset : http://archives.textfiles.com/stories.zip
# I got in in ./test_files/stories

# following tutorial from https://towardsdatascience.com/tf-idf-for-document-ranking-from-scratch-in-python-on-real-world-dataset-796d339a4089


import os
import re
#import urllib.request
#import zipfile
#import sys
from collections import namedtuple as nt

# file_names - dictionary of filenames
# text_title_tuples - dictionary { path : text_title_tuple }

class Dataset :
    _URL = "http://archives.textfiles.com/stories.zip"

    @property
    def file_names(self) :
        return self._fns

    @property
    def text_title_tuples(self) :
        return self._tts

    def __init__(self, relativePathToStories = 'test_files\\stories') :
        # next step - dataset will download stories.zip and with zipfile it'll extract all the needed files
        #path = os.path.basename(self._URL)
        #if not os.path.exists(path) :
        #    print("Downloading dataset {}...".format(path), file=sys.stderr)
        #    urllib.request.urlretrieve(self._URL, filename=path)
        
        folders = [x[0] for x in os.walk(os.path.join(str(os.getcwd()),relativePathToStories))]

        self._fns = {}
        self._tts = {} # text_title_tuple
        for folder in folders :
            with open(os.path.join(folder, 'index.html'), "r") as i_f :
                indices = i_f.read().strip()
            file_names = re.findall('><A HREF="(.*)">[^<]*</A> ', indices)
            file_titles = re.findall('<BR><TD> (.*)\n', indices)

            if len(file_names) != len(file_titles) :
                raise Exception("len(file_names) != len(file_titles)")

            ttt = nt("text_title_tuple", ["text", "title"])
            for file_name, title in zip(file_names, file_titles) :
                path = os.path.join(folder, file_name)
                self._fns[path] =  title
                with open(path, "r") as file :
                    self._tts[path] = ttt._make([file.read(), title])