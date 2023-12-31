import cv2
import requests
import numpy as np
from utils.imageloader import ImageLoader
import json

class PersonClassifier(ImageLoader):
    """A class for training and using a facial recognition model."""
    def __init__(self):
        super().__init__()
    
    def faceDetection(self, img):
        try:
            face_classifier = cv2.CascadeClassifier( cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            face = face_classifier.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(40,40))
            return face
        except Exception as e:
            print(f"Error loading cascade classifier: {e}")
            raise e

    def read_image_from_url(self, url,target_size=(600, 600)):
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
        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        img = cv2.fastNlMeansDenoising(img, None, h=10, searchWindowSize=21)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = cv2.resize(img, target_size)
        return img
    
    def load_images_from_folder(self, folder_path):
        """
        Load and preprocess images from a specified folder.

        Parameters:
        - folder_path (str): The path to the folder containing user images.

        Returns:
        - tuple: A tuple containing lists of images and corresponding labels.
        """
        images = []
        labels = []
        image_folders = super().OrganizePersonImages(folder_path)
        for folder in image_folders:
            for file in image_folders[folder]:
                image_path = file['secure_url']
                img = self.read_image_from_url(image_path)
                if img is not None:
                    images.append(img)
                    labels.append(file['label'])
        return images, labels
    
    def create_recognizer(self, images, labels):
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
            label_mapping = {user_id: idx for idx, user_id in enumerate(set(labels))}
            int_labels = [label_mapping[user_id] for user_id in labels]
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.train(images, np.array(int_labels))
            recognizer.save("trained_model.yml")
            # save the label mappings in a json file
            with open('label_mapping.json', 'w') as json_file:
                json.dump(label_mapping, json_file)

            return recognizer, label_mapping
        except Exception as e:
            print(f"Error creating recognizer: {e}")
            raise e
    
    def load_label_mapping(self):
        label_mapping = None
        with open("label_mapping.json", "r") as json_file:
            label_mapping = json.load(json_file)
        return label_mapping
            

    def load_recognizer(self):
        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read("trained_model.yml")
            return recognizer
        except Exception as e:
            print(f"Error loading recognizer: {e}")
            raise e

    def predictWithUrl(self, image_url, confidence_threshold=40):
        try:
            recognizer = self.load_recognizer()
            image = self.read_image_from_url(image_url)
            faces = self.faceDetection(image)
            # Check if a face is detected
            if len(faces) == 0:
                print("No face detected.")
                return None, None
            
            # Extract the first detected face (assuming there's only one)
            x, y, w, h = faces[0]
            face_roi = image[y:y+h, x:x+w]
            label, confidence = recognizer.predict(face_roi)

            # get label mappings
            label_mapping = self.load_label_mapping()
            if label_mapping is None:
                return None, None
            
            for labelx in label_mapping:
                if label_mapping[labelx] == label:
                    label = labelx

            # Check if the recognition confidence is above the threshold
            if confidence < confidence_threshold:
                # Recognition failed
                return None, None
            else:
                return label, confidence
        except Exception as e:
            print(f"Error predicting: {e}")
            raise e
    
    def predictWithImage(self, image, confidence_threshold=40):
        try:
            recognizer = self.load_recognizer()
            label, confidence = recognizer.predict(image)
            if confidence < confidence_threshold:
                # Recognition failed
                return None, None
            else:
                return label, confidence
        except Exception as e:
            print(f"Error predicting: {e}")
            raise e
    
    def getFaces(self, gray_img):
        try:
            faces = self.faceDetection(gray_img)
            if len(faces) == 0:
                raise Exception("No face detected")

            x, y, w, h = faces[0]
            face_roi = gray_img[y:y+h, x:x+w]
            # Additional pre-processing steps (e.g., Gaussian blur, denoising)
            face_roi = cv2.fastNlMeansDenoising(face_roi, None, h=10, searchWindowSize=21)
            face_roi = cv2.GaussianBlur(face_roi, (5, 5), 0)
            return face_roi
        except Exception as e:
            raise e

        
    def predictImageFile(self, imagedata, confidence_threshold=40, target_size=(600, 600)):
        try:
            recognizer = self.load_recognizer()
            nparray = np.frombuffer(imagedata, np.uint8)
            test_image = cv2.imdecode(nparray, cv2.IMREAD_GRAYSCALE)
            face_roi = self.getFaces(test_image)
            if face_roi is None:
                return None, None
            face_roi = cv2.resize(face_roi, target_size)
            
            # # Recognition using the pre-processed face
            label, confidence = recognizer.predict(face_roi)

            # get label mappings
            label_mapping = self.load_label_mapping()
            if label_mapping is None:
                return None, None
            
            for labelx in label_mapping:
                if label_mapping[labelx] == label:
                    label = labelx

            # Check if the recognition confidence is above the threshold
            if confidence < confidence_threshold:
                # Recognition failed
                return None, None
            else:
                return label, confidence

        except Exception as e:
            print(f"Error predicting: {e}")
            raise e