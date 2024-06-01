from utils.classifier import PersonClassifier
from flask import request, jsonify
import os
from utils.newcombined import PersonImageClassifier
from schema.schema import train_schema
from jsonschema import validate,ValidationError

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
        kr_predictions = predicted_people['predictions']

        return jsonify({"lbhprediction": lbhpredictions, "kr_predictions": kr_predictions}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400
        


def TrainClassifier():
    try:
        validate(schema=train_schema, instance=request.json)
        version = request.json['version']
        activation = request.json['activation']
        downloaded = 0
        if request.json['download'] == 1:
            downloaded = new_classifier._get_read_images("persons")
        train_ds, val_ds = new_classifier._load_local_dataset("persons", target_size=(224, 224), batch_size=32)
        classes = train_ds.class_names
        num_classes = len(classes)
        new_classifier._save_kr_labels(classes)
        new_classifier._train_lbh_model("persons")
        history = new_classifier._train_kr_model(num_classes, train_ds, val_ds, show_summary=True, epochs=10,
                version=version, activation=activation)
        eval_dict = new_classifier._display_evaluation(history)
        if request.json['remove'] == 1:
            new_classifier._remove_cloud_folder("persons")
        return jsonify({"msg":"Models Trained Successfully",
                        "evaluation": eval_dict, 
                        "itemsCount":downloaded, "modelName": new_classifier.krmodel}), 200
    except ValidationError as ve:
        return jsonify({"error":ve.message}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 400

def RealTimeDetection():
    try:
        new_classifier._realtime_detect(video_url=0)
        return jsonify({"msg": "Realtime detection ended successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

def CheckVariants():
    try:
        download = 1
        remove = 1
        save_folder_path = os.path.join(os.getcwd(), "persons")
        if os.path.exists(save_folder_path):
           download = 0
        return jsonify({"download": download, "remove": remove})
    except Exception as e:
        return jsonify({"error", str(e)}), 400