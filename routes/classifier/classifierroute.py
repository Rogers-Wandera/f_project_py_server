from flask import Blueprint
from controllers.classifier.classifiercontroller import TrainClassifier, startStream,stopStream,PredictPerson,CheckVariants
from middlewares.VerifyJwt import verifyjwt
from middlewares.VerifyRoles import verifyroles
from conn.rolelist import USER_ROLES
classifier_bp = Blueprint("main", __name__)

@classifier_bp.route("/predict", methods=["POST"])
@verifyjwt
@verifyroles(USER_ROLES['Admin'])
def PredictWithImage():
    return PredictPerson()

@classifier_bp.route("/train", methods=["POST"])
@verifyjwt
@verifyroles(USER_ROLES['Admin'])
def TrainModel():
    return TrainClassifier()

@classifier_bp.route("/realtime/start", methods=["POST"])
@verifyjwt
@verifyroles(USER_ROLES['Admin'])
def StartRealTimeDetection():
    return startStream()

@classifier_bp.route("/realtime/stop", methods=["POST"])
@verifyjwt
@verifyroles(USER_ROLES['Admin'])
def StopRealTimeDetection():
    return stopStream()

@classifier_bp.route("/variants", methods=["GET"])
@verifyjwt
@verifyroles(USER_ROLES['Programmer'])
def GetVariants():
    return CheckVariants()