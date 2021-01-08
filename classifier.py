from sklearn.svm import LinearSVC
from feature_extract import FeatureExtractor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from nltk.classify.scikitlearn import SklearnClassifier
from os.path import exists
import pickle
import tqdm
from sklearn.metrics import confusion_matrix

class Classifier:
    def __init__(self, trainNew, trainDirectory):
        self.data = {}
        self.labels = ["bearish", "bullish", "neutral"]
        self.trainSet = []
        self.testSet = []
        self.featureList = []
        self.featureSets = []
        self.trainDir = trainDirectory

        if not trainNew and exists("./data/pickles/model.pickle"):
            self.model = pickle.load(open('./data/pickles/model.pickle','rb'))
            
        else:
            print("Training model")
            self.model = SklearnClassifier(LinearSVC(random_state=0, tol=1e-5))
            self.load_data()
            self.extract_features()
            self.train_model()

    # end

        

    def load_data(self):
        
        self.data = {}
        for label in self.labels:
            file = open(self.trainDir + label + ".txt", 'r')
            for line in file:
                self.data[line] = label
        return self.data
    # end


    def extract_features(self):
        if(len(self.data) == 0):
            self.load_data()
        
        f = FeatureExtractor()
        counter = 0
        for tweet, label in tqdm.tqdm(self.data.items()):
            featureVector = f.get_feature_vector(tweet)
            if len(featureVector) > 0:
                self.featureSets.append((featureVector, label))         # dictionary of [bigrams in a tweet] : sentiment of that tweet
                self.featureList = self.featureList + featureVector     # list of all bigrams, later gets repeats removed
                self.trainSet.append((dict([(tuple(word), True) for word in featureVector]), label))
                counter += 1
        print(len(self.featureSets), "tweets total")
        self.featureList = list(set(tuple(i) for i in self.featureList))
        print(len(self.featureList), "unique features")
        print(len(self.trainSet), "training tweets")

        return self.trainSet
    # end


    def train_model(self):

        if(len(self.trainSet) == 0):
            self.extract_features()
        
        self.model.train(self.trainSet)

        pickle.dump(self.model, open("data/pickles/model.pickle", "wb"))

        return self.model
    # end

    def classify(self, tweetText):
        f = FeatureExtractor()
        tweetText = str(tweetText)
        featureVector = f.get_feature_vector(tweetText)
        features = dict([(tuple(word), True) for word in featureVector])

        prediction = self.model.classify(features)

        return prediction

