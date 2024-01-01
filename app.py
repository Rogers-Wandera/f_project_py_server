from flask import jsonify, g
from routes.approute import CreateApp
from flask_jwt_extended import JWTManager
from dotenv import load_dotenv
import os
from conn.connector import Connection
# from conn.config import config
from utils.imageloader import ImageLoader

load_dotenv()

app = CreateApp()
JWT_SECRET = os.getenv('JWT_SECRET')
imageuploader = ImageLoader()

app.config['JWT_TOKEN_LOCATION'] = ['headers']
app.config['JWT_SECRET_KEY'] = JWT_SECRET

jwt = JWTManager(app)

@app.before_request
def before_request():
    g.db = Connection()
    try:
        g.db.connect()
    except Exception as e:
        app.logger.error(f"Error connecting to the database: {str(e)}")
# @app.teardown_request
# def teardown_request(exception):
#     if hasattr(g, 'db'):
#         g.db.disconnect()

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