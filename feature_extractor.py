from sklearn.model_selection import train_test_split
from os import listdir
from os.path import isfile, join
from collections import OrderedDict
from sklearn.feature_selection import VarianceThreshold
import random

import lex

class BaseFeatureExtractor:
    def __init__(self, **kwargs):
        self.path_to_sources = kwargs['path_to_sources']
        self.cppfiles = self.get_cpp_files(self.path_to_sources)
        random.shuffle(self.cppfiles)
        self.training_data, self.test_data = self._train_test_split(self.cppfiles, kwargs.get('training_part', 0.5))
        self.scale_features = kwargs.get('scale', False)
        self.variance_threshold = kwargs.get('variance_threshold', False)
        
    def extract_features(self):
        self.X = [] 
        for f in self.training_data:
            self.X.append(self.get_features(self.path_to_sources + f))

        self.X_test = []
        for f in self.test_data:
                self.X_test.append(self.get_features(self.path_to_sources + f))

    def transform_features(self):
        if self.scale_features:
            self.scaler = StandardScaler()
            self.scaler.fit(self.X)
        
        if self.variance_threshold:
            self.sel = VarianceThreshold()
            self.sel.fit(self.X)            

        self.X = self.prepare_features(self.X)
        if len(self.X_test) > 0:
            self.X_test = self.prepare_features(self.X_test)
    

    def prepare_features(self, features):
        if self.scale_features:
            features = self.scale_features(features)
        if self.variance_threshold:
            features = self.variance_threshold_transform(features)

        return features

    def scale_features(self, features):
        return self.scaler.transform(features)

    def variance_threshold_transform(self, features):
        return self.sel.transform(features)

    def get_features(self):
        pass #todo: del func

    def _train_test_split(self, files, train_part):
        if train_part < 1:
            return train_test_split(files, train_size=train_part)
        else:
            return [files, list()] 

    def get_cpp_files(self, path):
        onlyfiles = [ f for f in listdir(path) if isfile(join(path, f)) ]
        return list(filter(lambda x : x[-3:] == "cpp", onlyfiles))


class NgramExtractor(BaseFeatureExtractor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n = kwargs.get('n', 1)
        self.train_model()
        self.extract_features()
        self.transform_features()

    def train_model(self):
        self.model = lex.get_ngram_model(self.path_to_sources, self.training_data, self.n)

    def get_features(self, filename):
        source = open(filename, 'r').read()
        ordered = OrderedDict(lex.get_ngram(source, self.model, self.n))
        return list(ordered.values())

class Word2VecExtractor(BaseFeatureExtractor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.features_size = kwargs.get("size", 20)
        self.train_model()
        self.extract_features()
        self.transform_features()

    def train_model(self):
        self.model = lex.get_word2vec_model(self.path_to_sources, self.training_data, self.features_size)

    def get_features(self, filename):
        ans = []
        tokens = lex.get_keywords(open(filename, "r").read())
        for t in tokens:
            if t in self.model:
                if len(ans) == 0:
                    ans = self.model.wv[t]
                else:
                    ans = ans + self.model.wv[t]
        return ans / len(tokens)
