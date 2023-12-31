from utils.classifier import PersonClassifier
from flask import request, jsonify
import os

ClassifierObj = PersonClassifier()

def Classifier():
   try:
        if not request.is_json:
            return jsonify({"error": "No Json data provided"}), 400
        
        json_data = request.get_json()
        if 'image' not in json_data:
            return jsonify({"error": "No image provided"}), 400
        
        imagedata = json_data['image']
        image_file_data = None
        if imagedata is None:
            return jsonify({"error": "No image provided"}), 400
        required_keys = ['filename', 'path']

        if not all(key in imagedata for key in required_keys):
            return jsonify({"error": "Missing required keys"}), 400
        
        image_path = os.path.join(imagedata['destination'], imagedata['filename'])
        if not os.path.exists(image_path):
            return jsonify({"error": "Image file not found"}), 404
        with open(image_path, 'rb') as image_file:
            image_file_data = image_file.read()
        
        if image_file_data is None:
            return jsonify({"error": "Failed to read image file"}), 400
        
        label, confidence = ClassifierObj.predictImageFile(image_file_data)
        return jsonify({"label": label, "confidence": confidence}), 200

   except Exception as e:
        return jsonify({"error": str(e)}), 400

def TrainClassifier():
   try:
        images,labels = ClassifierObj.load_images_from_folder("persons")
        ClassifierObj.create_recognizer(images, labels)
        return jsonify({"msg": "Model Trained successfully"}), 200
   except Exception as e:
    return jsonify({"error": str(e)}), 400
