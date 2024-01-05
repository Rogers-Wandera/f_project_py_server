from flask import jsonify, g
from routes.approute import CreateApp
from flask_jwt_extended import JWTManager
from dotenv import load_dotenv
import os
# from conn.config import config
from utils.imageloader import ImageLoader
# from utils.personaudio import PersonAudio

load_dotenv()

# obj = PersonAudio(label_path="personsaudiolabels", model_path="personsaudiomodel")
# pathurl = r"C:\Users\Rogers\Downloads\WhatsApp Unknown 2024-01-05 at 11.48.12 AM\test.ogg"
# results = obj._predict_person_audio(pathurl)
# print(results)
# obj._train_person_audio_model("personsaudio", type="cloudinary")
# X_train, X_test, y_train, y_test=obj._load_cloudinary_dataset("personsaudio", max_results=500)

app = CreateApp()
JWT_SECRET = os.getenv('JWT_SECRET')
imageuploader = ImageLoader()

app.config['JWT_TOKEN_LOCATION'] = ['headers']
app.config['JWT_SECRET_KEY'] = JWT_SECRET

jwt = JWTManager(app)

# @app.before_request
# def before_request():

@app.errorhandler(404)
def not_found_error(error):
    errordata = {"error": "Page not found"}
    return jsonify(errordata), 404

@app.route('/')
def index():
    image_loader = ImageLoader()
    folder_path = "persons"
    
    try:
        images = image_loader.OrganizePersonImages(folder_path)
        print(images)
        return jsonify(images)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)