from sklearn.svm import LinearSVC

class Train_svm:
    def __init__(self):
        self.data = {}
        self.labels = ["bearish", "bullish", "neutral"]
        self.trainSet = []
        self.featureList = []
        self.featuresets = []
    # end

    def load_data(self):
        self.data = {}
        for label in self.labels:
            file = open("./data/train/" + label + ".txt", 'r')
            for line in file:
                self.data[line] = label
        return self.data



        
if __name__ == "__main__":
    t = Train_svm()
    
    print(t.load_data())