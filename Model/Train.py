from anago.utils import load_data_and_labels
import anago
import os


class NER:

    def __init__(self):
        self.ROOT_DIR = os.getcwd()
        self.x_train = self.y_train = []
        self.x_valid = self.y_valid = []
        self.x_test = self.y_test = []
        self.model = anago.Sequence()

    def load_data(self):
        self.x_train, self.y_train = load_data_and_labels(self.ROOT_DIR + '/Dataset/English/train.txt')
        self.x_valid, self.y_valid = load_data_and_labels(self.ROOT_DIR + '/Dataset/English/valid.txt')
        self.x_test, self.y_test = load_data_and_labels(self.ROOT_DIR + '/Dataset/English/test.txt')

    def save_model(self):
        self.model.save(self.ROOT_DIR + '/Model/model/weights.h5',
                        self.ROOT_DIR + '/Model/model/params.json',
                        self.ROOT_DIR + '/Model/model/preprocessor.pickle')

    def load_model(self):
        self.model = self.model.load(self.ROOT_DIR + '/Model/model/weights.h5',
                                     self.ROOT_DIR + '/Model/model/params.json',
                                     self.ROOT_DIR + '/Model/model/preprocessor.pickle')

    def fit(self):
        self.model.fit(self.x_train, self.y_train, self.x_valid, self.y_valid, epochs=15)

    def test_model(self):
        self.model.score(self.x_test, self.y_test)

    def predict(self, s):
        return self.model.analyze(s)
