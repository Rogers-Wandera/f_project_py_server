from flask import Blueprint
from controllers.classifier.classifiercontroller import PredictWithLocalImage, TrainClassifier, RealTimeDetection,PredictWithImageUrl
from middlewares.VerifyJwt import verifyjwt
from middlewares.VerifyRoles import verifyroles
from conn.rolelist import USER_ROLES
classifier_bp = Blueprint("main", __name__)

@classifier_bp.route("/local", methods=["POST"])
@verifyjwt
@verifyroles(USER_ROLES['Admin'])
def PredictWithImage():
    return PredictWithLocalImage()

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

@classifier_bp.route("/url", methods=["POST"])
@verifyjwt
@verifyroles(USER_ROLES['Admin'])
def PredictImageWithUrl():
    return PredictWithImageUrl()