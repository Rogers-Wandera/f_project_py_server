from flask import Flask
from routes.classifier.classifierroute import classifier_bp
from routes.classifier.audioroute import audioblueprint
def CreateApp():
    app = Flask(__name__)
    app.register_blueprint(classifier_bp, url_prefix='/classifier')
    app.register_blueprint(audioblueprint, url_prefix='/audio')
    return app