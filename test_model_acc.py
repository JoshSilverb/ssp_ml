


from classifiers.classifier import Classifier


if __name__ == "__main__":
    labels = ["bearish", "bullish", "neutral"]
    results = {"bearish": {True: 0, False: 0},
               "bullish": {True: 0, False: 0},
               "neutral": {True: 0, False: 0}}

    classifier = Classifier("clstm", False, "./data/train/")

    for label in labels:
        print("Testing label", label)
        with open(f"./data/train/{label}.txt", 'r') as testFile:
            for tweet in testFile:
                classification = classifier.classify(tweet)
                results[label][classification == label] += 1
    
    print(results)
    num_correct = results["bearish"][True] + results["bullish"][True] + results["neutral"][True]
    num_wrong = results["bearish"][False] + results["bullish"][False] + results["neutral"][False]
    print("Accuracy:", num_correct/(num_correct+num_wrong))