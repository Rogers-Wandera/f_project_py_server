from utils.classifier import PersonClassifier
from flask import request, jsonify
import os
from utils.newcombined import PersonImageClassifier

ClassifierObj = PersonClassifier()
new_classifier = PersonImageClassifier("lhb_person_model", "kr_person_model", input_shape=(224,224, 3), target_size=(224, 224))

def get_local_image(json_data):
    try:
        imagedata = json_data['image']
        image_file_data = None
        if imagedata is None:
            raise Exception("No image provided")
        required_keys = ['filename', 'path']

        if not all(key in imagedata for key in required_keys):
            raise Exception("Missing required keys")
        
        image_path = os.path.join(imagedata['destination'], imagedata['filename'])
        if not os.path.exists(image_path):
            raise Exception("Image file not found")
        
        image_file_data = image_path
        
        if image_file_data is None:
            raise ("Failed to read image file")
        
        return image_file_data
    except Exception as e:
        raise e

def read_image_from_url(json_data):
    try:
        imagedata = json_data['image']
        image_file_data = None
        if imagedata is None:
            raise Exception("No image provided")
        required_keys = ['url']

        if not all(key in imagedata for key in required_keys):
           raise Exception("Missing required keys")
        
        image_file_data = imagedata['url']

        if image_file_data is None:
            raise ("Failed to read image file")
        return image_file_data
    except Exception as e:
        raise e

def PredictWithLocalImage():
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
        
        width = None
        height = None
        
        if "width" in json_data and "height" in json_data:
            width = json_data['width']
            height = json_data['height']
        
        if width is None or height is None:
            faces_list = ClassifierObj.predictImageFile(image_file_data)
            return jsonify(faces_list), 200
        else:
            faces_list = ClassifierObj.predictImageFile(image_file_data,40, (width, height))
            return jsonify(faces_list), 200

   except Exception as e:
        return jsonify({"error": str(e)}), 400


def PredictWithImageUrl():
        if not request.is_json:
            return jsonify({"error": "No Json data provided"}), 400
        
        json_data = request.get_json()
        if 'image' not in json_data:
            return jsonify({"error": "No image provided"}), 400
        
        imagedata = json_data['image']
        image_file_data = None
        if imagedata is None:
            return jsonify({"error": "No image provided"}), 400
        required_keys = ['url']

        if not all(key in imagedata for key in required_keys):
            return jsonify({"error": "Missing required keys"}), 400
        
        image_file_data = ClassifierObj.read_image_from_url(imagedata['url'])
        if image_file_data is None:
            return jsonify({"error": "Failed to read image file"}), 400
        
        # predicting the image
        label, confidence = ClassifierObj.predictWithUrl(imagedata['url'])
        return jsonify({"label": label, "confidence": confidence}), 200


def PredictPerson():
    try:
        accepted_types = ["url_image", "local_image"]
        if not request.json:
            return jsonify({"error": "No Json data provided"}), 400
        json_data = request.json
        if "type" not in json_data:
            return jsonify({"error": "No type provided"}), 400
        if json_data['type'] not in accepted_types:
            return jsonify({"error": "Invalid type accepted types are url_image, local_image"}), 400
        if "image" not in json_data:
            return jsonify({"error": "No image provided"}), 400
        image_file_data = None
        if json_data['type'] == "url_image":
            image_file_data = read_image_from_url(json_data)
        else:
            image_file_data = get_local_image(json_data)
        
        if image_file_data is None:
            return jsonify({"error": "Failed to read image file"}), 400
        
        predictions = new_classifier._predict_image(image_file_data)
        predicted_people = new_classifier.show_predicted_people(predictions['lbhpredictions'], predictions['krpredictions'])
        lbhpredictions = predicted_people['lbhprediction']

        return jsonify({"lbhprediction": lbhpredictions }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400
        


def TrainClassifier():
   try:
        downloaded = new_classifier._get_read_images("persons")
        train_ds, val_ds = new_classifier._load_local_dataset("persons", target_size=(224, 224), batch_size=32)
        classes = train_ds.class_names
        num_classes = len(classes)
        new_classifier._save_kr_labels(classes)
        lbh_model = new_classifier._train_lbh_model("persons")
        history = new_classifier._train_kr_model(num_classes, train_ds, val_ds, show_summary=True, epochs=10)
        eval_dict = new_classifier._display_evaluation(history)
        removed = new_classifier._remove_cloud_folder("persons")
        return jsonify({"msg":"Models Trained Successfully",
                        "kr_evaluation": eval_dict, 
                        "downloaded":downloaded, 
                        "removed": removed, "lbh_model": lbh_model[1]}), 200
   except Exception as e:
    return jsonify({"error": str(e)}), 400

def RealTimeDetection():
    try:
        ClassifierObj.create_real_time_detection()
        return jsonify({"msg": "Realtime detection ended successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400
