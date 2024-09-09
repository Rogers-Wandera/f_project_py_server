from flask import jsonify, render_template, Flask
from routes.approute import CreateApp
from flask_jwt_extended import JWTManager
from dotenv import load_dotenv
import os
from flask_socketio import SocketIO
from flask_cors import CORS
import threading
import cv2
from sockets.sockets import socketinstance
from utils.lbhclassifier import PersonLBHClassifier

load_dotenv()
app = Flask(__name__)
app = CreateApp(app)
CORS(app, resources={r"/socket.io/*": {"origins": "http://localhost:5173"}})

JWT_SECRET = os.getenv('JWT_SECRET')
app.config['JWT_TOKEN_LOCATION'] = ['headers']
app.config['JWT_SECRET_KEY'] = JWT_SECRET

socket = SocketIO(app=app, cors_allowed_origins=["http://localhost:5173", "http://localhost:5000"])
sio = socketinstance.initialize()

pcl = PersonLBHClassifier()

video_stream = False

jwt = JWTManager(app)

@app.errorhandler(404)
def not_found_error(error):
    errordata = {"error": "Page not found"}
    return jsonify(errordata), 404

@socket.on("connect")
def SocketConnection():
    print("a connection established")

@sio.event
def connect():
    print('Connected to the server')

def capture_and_stream():
    global video_stream
    video_capture = cv2.VideoCapture(0)

    while video_stream:
        ret, frame = video_capture.read()

        if not ret:
            break
        pcl._realtime_detection("lhb_person_model", frame=frame, socket=socket, userId=1)
    video_capture.release()

@socket.on("startvideo")
def StartVideo():
    global video_stream
    if not video_stream:
        video_stream = True
        threading.Thread(target=capture_and_stream).start()

@socket.on("stopvideo")
def Stop_Video():
    global video_stream
    video_stream = False

sio.connect("http://localhost:3500/")

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)