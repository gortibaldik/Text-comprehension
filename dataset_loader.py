# loader of custom data from dataset : http://archives.textfiles.com/stories.zip
# I got in in ./test_files/stories

# following tutorial from https://towardsdatascience.com/tf-idf-for-document-ranking-from-scratch-in-python-on-real-world-dataset-796d339a4089


import os
import re
import urllib.request
import zipfile
import sys
from pathlib import Path, PurePosixPath
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
        path = os.path.basename(self._URL)
        if not os.path.exists(path) :
            print("Downloading dataset {}...".format(path), file=sys.stderr)
            urllib.request.urlretrieve(self._URL, filename=path)
        
        with zipfile.ZipFile(path, "r") as zip_file :
            index_files = [ file_name for file_name in zip_file.namelist() if os.path.basename(file_name) == "index.html"]

            self._fns = {}
            self._tts = {} # text_title_tuple
            for index_file in index_files :
                with zip_file.open(index_file, "r") as i_f :
                    indices = i_f.read().decode("utf-8").strip()

                input("read indices from {}".format(index_file))
                file_names = re.findall('><A HREF="(.*)">[^<]*</A> ', indices)
                file_titles = re.findall('<BR><TD> (.*)\n', indices)

                if len(file_names) != len(file_titles) :
                    raise Exception("len(file_names) != len(file_titles)")

                ttt = nt("text_title_tuple", ["text", "title"])
                path_index_file = Path(index_file)
                folder = path_index_file.parents[0]
                print("current folder : {}".format(folder))
                for file_name, title in zip(file_names, file_titles) :
                    path = PurePosixPath(folder / file_name)
                    self._fns[path] =  title
                    try :
                        with zip_file.open(str(path), "r") as file :
                            to_insert = ttt._make([file.read().decode("iso-8859-1"), title])
                    except :
                        print("ERROR IN DECODING {}".format(path))

                    self._tts[str(path)] = to_insert