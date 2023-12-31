from flask import Flask
from routes.classifier.classifierroute import classifier_bp
def CreateApp():
    app = Flask(__name__)
    app.register_blueprint(classifier_bp, url_prefix='/classifier')
    return app