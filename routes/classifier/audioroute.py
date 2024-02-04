from middlewares.VerifyJwt import verifyjwt
from middlewares.VerifyRoles import verifyroles
from controllers.audio.audioclassifier import TrainAudioClassifier, PredictAudio
from conn.rolelist import USER_ROLES
from flask import Blueprint

audioblueprint = Blueprint("audio", __name__)

@audioblueprint.route("/train", methods=["POST"])
@verifyjwt
@verifyroles(USER_ROLES['Admin'])
def TrainAudioModel():
    return TrainAudioClassifier()

@audioblueprint.route("/predict", methods=["POST"])
@verifyjwt
@verifyroles(USER_ROLES['Admin'])
def PredictPersonAudio():
    return PredictAudio()