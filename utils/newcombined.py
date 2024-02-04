import cv2
from utils.lbhclassifier import PersonLBHClassifier
import json
from conn.connector import Connection
from utils.kerasclassifier import ImagePersonClassifier
import os
from keras.losses import SparseCategoricalCrossentropy

dbconnect = Connection()

class PersonImageClassifier(PersonLBHClassifier, ImagePersonClassifier):
    def __init__(self, lbhmodelname, krmodelname, input_shape=(224,224, 3), target_size=(224, 224)):
        super().__init__()
        self.lbhmodel = lbhmodelname
        self.krmodel = krmodelname
        self.input_shape = input_shape
        self.target_size = target_size
    
    def _save_label_mappings(self, label_mappings, file_path):
        try:
            with open(file_path, 'w') as json_file:
                json.dump(label_mappings, json_file)
        except Exception as e:
            raise e
    def _read_image(self, image_file):
        try:
            image = None
            #check if image_file contains htpp or https
            if image_file.startswith("http"):
               image = self._read_image_from_url(image_file)
            else:
                image = cv2.imread(image_file)
                image = cv2.resize(image, self.target_size)
            return image
        except Exception as e:
            raise e
    def _predict_image(self, image):
        try:
            image = self._read_image(image)
            if image is None:
                raise Exception("Image not found")
            
            lbhpredictions = self._predict_image_lbh(image,self.lbhmodel)
            krpredictions = self._predict_with_image_kr(image,self.krmodel)
            return {"lbhpredictions": lbhpredictions, "krpredictions": krpredictions}
        except Exception as e:
            raise e
    
    def _train_lbh_model(self, folder_path):
        try:
            images, labels = self._load_images_from_folder(folder_path, self.target_size)
            results = self.create_recognizer(images, labels, self.lbhmodel)
            return results
        except Exception as e:
            raise e
    def _train_kr_model(self, num_classes,train_ds, test_ds, version="v1", activation="relu",
                        optimizer="adam", loss=SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"],
                       show_summary=True, epochs=10):
       try:
            model = None
            if version == "vl":
                model = self._create_model_v1(num_classes, input_shape=(self.input_shape), activation=activation)
            else:
               model = self._create_model_v2(num_classes, input_shape=(self.input_shape), activation=activation)
            
            if model is None:
                raise Exception("Model not found")
            model = self._compile_model(model, optimizer=optimizer, loss=loss, metrics=metrics, show_summary=show_summary)
            history = self._train_model(model, train_ds, test_ds, self.krmodel, epochs=epochs)
            return history
       
       except Exception as e:
           raise e
    
    def _save_kr_labels(self, labels):
        try:
            label_path = os.path.join(os.getcwd(), "models", "labels", f"{self.krmodel}_labels.json")
            self._save_label_mappings(labels, label_path)
        except Exception as e:
            raise e
    
    def _get_kr_labels(self):
        try:
            label_mapping = None
            with open(f"models/labels/{self.krmodel}_labels.json", "r") as json_file:
                label_mapping = json.load(json_file)
            return label_mapping
        except Exception as e:
            raise e
        
    def show_predicted_people(self, lbhprediction, krprediction):
        try:
            lbhpredictions = self._show_predicted_personlbh(lbhprediction, self.lbhmodel)
            krpredict = self._predicted_class(krprediction)
            kr_labels = self._get_kr_labels()
            lr_predicted_class = self._show_predicted_class(krprediction, kr_labels)
            krpredictions = self._show_predicted_people_kr(krpredict, kr_labels)
            return {"lbhprediction": lbhpredictions, "predictions": {"predicted_class": lr_predicted_class, "predicted_people": krpredictions}}
        except Exception as e:
            print(f"Error showing predicted people: {e}")
            raise e
    
    def _realtime_detect(self, video_url=0):
        try:
            return self._realtime_detection(modalname=self.lbhmodel, video_url=video_url)
        except Exception as e:
            raise e
