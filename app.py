from flask import jsonify, g
from routes.approute import CreateApp
from flask_jwt_extended import JWTManager
from dotenv import load_dotenv
import os

load_dotenv()

app = CreateApp()
JWT_SECRET = os.getenv('JWT_SECRET')
app.config['JWT_TOKEN_LOCATION'] = ['headers']
app.config['JWT_SECRET_KEY'] = JWT_SECRET

jwt = JWTManager(app)

@app.errorhandler(404)
def not_found_error(error):
    errordata = {"error": "Page not found"}
    return jsonify(errordata), 404


@app.route('/')
def index():
    jsonify({"msg": "Welcome to the app"}), 200

if __name__ == '__main__':
    app.run(debug=True)
