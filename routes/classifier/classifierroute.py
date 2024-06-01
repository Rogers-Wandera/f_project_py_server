from flask import Blueprint
from controllers.classifier.classifiercontroller import TrainClassifier, RealTimeDetection,PredictPerson,CheckVariants
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

@classifier_bp.route("/realtime", methods=["GET"])
@verifyjwt
@verifyroles(USER_ROLES['Admin'])
def GetRealTimeDetection():
    return RealTimeDetection()

@classifier_bp.route("/variants", methods=["GET"])
@verifyjwt
@verifyroles(USER_ROLES['Programmer'])
def GetVariants():
    return CheckVariants()