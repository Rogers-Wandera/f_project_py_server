from flask import Blueprint
from controllers.classifier.classifiercontroller import Classifier as ClassifierController, TrainClassifier

classifier_bp = Blueprint("main", __name__)
@classifier_bp.route("/", methods=["POST"])
def Classifier():
    return ClassifierController()

@classifier_bp.route("/train", methods=["POST"])
def TrainModel():
    return TrainClassifier()