from routes.classifier.classifierroute import classifier_bp
from routes.classifier.audioroute import audioblueprint
def CreateApp(app):
    app.register_blueprint(classifier_bp, url_prefix='/classifier')
    app.register_blueprint(audioblueprint, url_prefix='/audio')
    return app