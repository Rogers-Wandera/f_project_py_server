import cv2
import requests
import numpy as np
from utils.imageloader import ImageLoader
import json
from conn.connector import Connection
# from utils.newclassifier import ImagePersonClassifier
import os

dbconnect = Connection()

class PersonLBHClassifier(ImageLoader):
    def _read_image_from_url(self, url, target_size=(224, 224)):
        """
        Read an image from a URL, convert it to grayscale, and resize it.

        Parameters:
        - url (str): The URL of the image.
        - target_size (tuple): The target size for the image after resizing.

        Returns:
        - np.array: The processed image.
        """
        response = requests.get(url)
        img_array = np.array(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.fastNlMeansDenoising(
            img, None, h=10, templateWindowSize=5, searchWindowSize=21)
        img = cv2.resize(img, target_size)
        return img
    
    def load_label_mapping(self, modalname):
        try:
            label_mapping = None
            with open(f"models/labels/{modalname}_labels.json", "r") as json_file:
                label_mapping = json.load(json_file)
            return label_mapping
        except Exception as e:
            raise e

    def _load_images_from_folder(self, folder_path, target_size):
        try:
            folder_path_exist = os.path.join(os.getcwd(), folder_path)
            DIR = folder_path_exist
            if not os.path.exists(folder_path_exist):
                raise Exception("Image folder not found")
            images = []
            labels = []
            folder_path_exist = os.listdir(folder_path_exist)
            for folder in folder_path_exist:
                image_folder = os.path.join(DIR, folder)
                for file in os.listdir(image_folder):
                    image_file = os.path.join(image_folder, file)
                    img = cv2.imread(image_file)
                    if img is not None:
                        faces = self._face_detection(img)
                        if len(faces) == 0:
                            continue
                        for face in faces:
                            face = cv2.resize(face, target_size)
                            face = cv2.fastNlMeansDenoising(
                            face, None, h=10, templateWindowSize=5, searchWindowSize=21)
                            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                            # cv2.imshow("image", face)
                            # cv2.waitKey(0)
                            images.append(face)
                            labels.append(folder)
            return images, labels
        except Exception as e:
            print(f"Error loading images from folder: {e}")
            raise e
    
    def create_recognizer(self, images, labels,modalname):
        """
        Train a facial recognition model.

        Parameters:
        - images (list): List of images for training.
        - labels (list): List of corresponding labels.

        Returns:
        - tuple: A tuple containing the trained recognizer and label mapping.
        """
        try:
            # i create numerical mappings for the labels
            label_mapping = {user_id: idx for idx,
                             user_id in enumerate(set(labels))}
            int_labels = [label_mapping[user_id] for user_id in labels]
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.train(images, np.array(int_labels))
            recognizer.save(f"models/models/{modalname}.yml")
            # save the label mappings in a json file
            with open(f'models/labels/{modalname}_labels.json', 'w') as json_file:
                json.dump(label_mapping, json_file)

            return recognizer, label_mapping
        except Exception as e:
            print(f"Error creating recognizer: {e}")
            raise e
    
    def _load_recognizer(self, modalname):
        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read("models/models/"+modalname+".yml")
            return recognizer
        except Exception as e:
            print(f"Error loading recognizer: {e}")
            raise e
    
    def _predict_image_lbh(self, image, modalname, target_size=(224, 224)):
        try:
            recognizer = self._load_recognizer(modalname)
            faces = self._face_detection(image)
            prediction = []
            # Check if a face is detected
            if len(faces) == 0:
                return []
            
            for face in faces:
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face = cv2.resize(face, target_size)
                label, confidence = recognizer.predict(face)
                prediction.append({"label": label, "confidence": int(round(confidence))})
            return prediction
        except Exception as e:
            print(f"Error predicting: {e}")
            raise e
    
    def _show_predicted_personlbh(self, prediction, modalname):
        try:
            if len(prediction) == 0:
               return []
            predicted = []
            label_mapping = self.load_label_mapping(modalname)
            if label_mapping is None:
                return []
            for person in prediction:
                label = person["label"]
                confidence = person["confidence"]
                for labelx in label_mapping:
                    if label_mapping[labelx] == label:
                        user = dbconnect.findone("person",{"id": labelx, "isActive": 1})
                        if user != None:
                            user_name = f"{user['firstName']} {user['lastName']}"
                            predicted.append({"id": labelx, "confidence": confidence,"label":user_name})
            return predicted
        except Exception as e:
            raise e
    def detect_bounding_box(self, vid):
        faces = self._bounding_boxes(vid)
        for face in faces:
            x1, y1, x2, y2 = face
            cv2.rectangle(vid, (x1, y1), (x2, y2), (0, 255, 0), 4)
        return faces
    
    def _realtime_detection(self, modalname, video_url=0):
        try:
            video_capture = cv2.VideoCapture(video_url)
            video_stream = True
            label_mapping = self.load_label_mapping(modalname)
            recognizer = self._load_recognizer(modalname)
            while video_stream:
                results, video_frame = video_capture.read()
                if results is False:
                    break
                faces = self.detect_bounding_box(video_frame)
                for face in faces:
                    face_roi = video_frame[face[1]: face[1] + face[3], face[0]: face[0] + face[2]]
                    gray_image = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                    label, confidence = recognizer.predict(gray_image)
                    user = None
                    nameprefix = None
                    for labelx in label_mapping:
                        if label_mapping[labelx] == label:
                            userdata = dbconnect.findone("person", {"id": labelx, "isActive": 1})
                            if userdata is not None:
                                user = f"{userdata['firstName']} {userdata['lastName']}"
                                nameprefix = f"{userdata['firstName'][0]}.{userdata['lastName'][0]}"
                            label = labelx
                    prefix = "Unknown" if user is None else nameprefix
                    text = f"{prefix}: {confidence:.2f}"
                    cv2.putText(video_frame, text, (face[0], face[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv2.imshow("Video Frame", video_frame)
                    key = cv2.waitKey(1)
                    if key in [ord('q'), 27, 255]:
                        video_stream = False
            video_capture.release()
            cv2.destroyAllWindows()

            if not video_stream:
                print("Video stream ended.")
        except Exception as e:
            raise e