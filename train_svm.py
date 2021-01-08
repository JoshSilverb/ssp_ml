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
    def __init__(self):
        self.data = {}
        self.labels = ["bearish", "bullish", "neutral"]
        self.trainSet = []
        self.testSet = []
        self.featureList = []
        self.featureSets = []
        self.model = SklearnClassifier(LinearSVC(random_state=0, tol=1e-5, verbose=1))
    # end

    def check_for_pickles(self):

        c = False
        '''
        if exists("./data/pickles/featureList.pickle"):
            self.featureList = pickle.load(open('./data/pickles/featureList.pickle','rb'))
            print("featureList loaded")
            a = True
        if exists("./data/pickles/featureSets.pickle"):
            self.featureList = pickle.load(open('./data/pickles/featureSets.pickle','rb'))
            print("featureSets loaded")
            b = True
        '''
        if exists("./data/pickles/trainSet.pickle"):
            self.trainSet = pickle.load(open('./data/pickles/trainSet.pickle','rb'))
            print(self.trainSet)
            print("trainSet loaded")
            c = True
        return c
        
        

    def load_data(self):
        
        self.data = {}
        for label in self.labels:
            file = open("./data/train/" + label + ".txt", 'r')
            for line in file:

                self.data[line] = label
        return self.data
    # end


    def extract_features(self):
        if(len(self.data) == 0):
            self.load_data()
        
        f = FeatureExtractor()
        print("getting featureVectors")
        counter = 0
        for tweet, label in tqdm.tqdm(self.data.items()):
            featureVector = f.get_feature_vector(tweet)
            if len(featureVector) > 0:
                self.featureSets.append((featureVector, label))         # dictionary of [bigrams in a tweet] : sentiment of that tweet
                self.featureList = self.featureList + featureVector     # list of all bigrams, later gets repeats removed
                #if counter % 5 == 0:    # 80/20 train/test split
                #    self.testSet.append((dict([(tuple(word), True) for word in featureVector]), label))
                #else:
                self.trainSet.append((dict([(tuple(word), True) for word in featureVector]), label))
                counter += 1
        print(len(self.featureSets), "tweets total")
        self.featureList = list(set(tuple(i) for i in self.featureList))
        print(len(self.featureList), "unique features")
        print(len(self.trainSet), "training tweets")
        print(len(self.testSet), "testing tweets")

        # use pickle to store data
        pickle.dump(self.featureList, open("data/pickles/featureList.pickle", "wb"))
        pickle.dump(self.featureSets, open("data/pickles/featureSets.pickle", "wb"))
        # pickle.dump(self.featureSets, open("data/pickles/trainSet.pickle", "wb"))
        

        return self.trainSet
    # end


    def train_model(self):
        
        if exists("./data/pickles/model.pickle"):
            self.model = pickle.load(open('./data/pickles/model.pickle','rb'))
            print("model loaded from saved file")
            return self.model

        if(len(self.trainSet) == 0):
            self.extract_features()
        
        print("Beginning Linear SVC training...")
        self.model.train(self.trainSet)
        print("\nDone")

        print("Saving model as pickle...")
        pickle.dump(self.model, open("data/pickles/model.pickle", "wb"))
        print("Done")

        return self.model
    # end

    def classify(self, tweetText):
        f = FeatureExtractor()
        print("Processing tweet...")
        tweetText = str(tweetText)
        featureVector = f.get_feature_vector(tweetText)
        features = dict([(tuple(word), True) for word in featureVector])
        print("Done")

        print("Predicting label")
        prediction = self.model.classify(features)
        print("done")
        return prediction

    


        
if __name__ == "__main__":
    t = Classifier()
    #saved = t.check_for_pickles()
    #if not saved:
    t.load_data()
    t.extract_features()
    print(t.trainSet[0])
    t.train_model()
    predicted = []
    real = []
    print("Testing prediction on test data...")
    for features, label in tqdm.tqdm(t.testSet):
        predicted.append(t.model.classify(features))
        real.append(label)
    print("Done")
    print("Confusion matrix:")
    print(confusion_matrix(real, predicted, labels=t.labels))  