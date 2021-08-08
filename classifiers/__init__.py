from classifiers.linearSvcClassifier import SvcClassifier
from classifiers.clstmClassifier import ClstmClassifier
from classifiers.cnnClassifier import ConvClassifier


VALID_MODEL_NAMES = ("svc", "cnn", "clstm")
NAME_TO_MODEL_MAP = {"svc": SvcClassifier(), "cnn": ConvClassifier(), "clstm": ClstmClassifier()}