import cv2
import requests
import numpy as np
from utils.imageloader import ImageLoader
import json
from conn.connector import Connection

dbconnect = Connection()
class PersonClassifier(ImageLoader):
    """A class for training and using a facial recognition model."""
    def __init__(self):
        super().__init__()
    
    def faceDetection(self, img):
        try:
            face_classifier = cv2.CascadeClassifier( cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            face = face_classifier.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)
            return face
        except Exception as e:
            print(f"Error loading cascade classifier: {e}")
            raise e

    def read_image_from_url(self, url,target_size=(400, 500)):
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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.fastNlMeansDenoising(img, None, h=10, templateWindowSize=5, searchWindowSize=21)
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
                    faces = self.faceDetection(img)
                    if len(faces) == 0:
                       continue
                    x, y, w, h = faces[0]
                    img = img[y:y+h, x:x+w]
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
        
    def getlabeluser(self, label):
        try:
            # get all persons from db
            results = dbconnect.findall("person")
            row = None
            for result in results:
                if result['id'] == label:
                    row = result
            return row
        except Exception as e:
            raise e
    
    def load_label_mapping(self):
        try:
            label_mapping = None
            with open("label_mapping.json", "r") as json_file:
                label_mapping = json.load(json_file)
            return label_mapping
        except Exception as e:
            raise e
            

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
    
    def getFaces(self, gray_img,target_size):
        try:
            img = cv2.fastNlMeansDenoising(gray_img, None, h=10, templateWindowSize=5, searchWindowSize=21)
            img = cv2.resize(img, target_size)
            faces = self.faceDetection(img)
            if len(faces) == 0:
                raise Exception("No face detected")
            newfaces = []
            for (x, y, w, h) in faces:
                face_roi = img[y:y+h, x:x+w]
                newfaces.append(face_roi)
            return newfaces
        except Exception as e:
            raise e

        
    def predictImageFile(self, imagedata, confidence_threshold=40, target_size=(400, 500)):
        try:
            recognizer = self.load_recognizer()
            nparray = np.frombuffer(imagedata, np.uint8)
            test_image = cv2.imdecode(nparray, cv2.IMREAD_GRAYSCALE)
            face_roi = self.getFaces(test_image, target_size)
            if face_roi is None:
                return None, None,
            
            faces_list = []
            # # Recognition using the pre-processed face
            for face in face_roi:
                label, confidence = recognizer.predict(face)
                # get label mappings
                label_mapping = self.load_label_mapping()
                if label_mapping is None:
                    return None, None
                
                for labelx in label_mapping:
                    if label_mapping[labelx] == label:
                        label = labelx
                faces_list.append({"label": label, "confidence": confidence})
            return faces_list

        except Exception as e:
            print(f"Error predicting: {e}")
            raise e
        finally:
            dbconnect.disconnect()
    
    def detect_bounding_box(self, vid):
        gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
        faces = self.faceDetection(gray_image)
        for (x,y,w,h) in faces:
            cv2.rectangle(vid, (x,y), (x+w,y+h), (0,255,0), 4)
        return faces
    
    def create_real_time_detection(self, video_url=0):
        try:
            video_capture= cv2.VideoCapture(video_url)
            recognizer = self.load_recognizer()
            video_stream = True
            while video_stream:
                results, video_frame = video_capture.read()
                if results is False:
                    break #break out of the video capture loop

                faces = self.detect_bounding_box(video_frame) # call the detect_bounding_box function
                for (x,y,w,h) in faces:
                    face_roi = video_frame[y:y+h, x:x+w]
                    face_roi = cv2.fastNlMeansDenoising(face_roi, None, h=10,templateWindowSize=5, searchWindowSize=21)
                    # face_roi = cv2.GaussianBlur(face_roi, (5, 5), 0)
                    # turn image in gray scale
                    gray_image = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                    label, confidence = recognizer.predict(gray_image)
                    label_mapping = self.load_label_mapping()
                    user = None
                    nameprefix = None
                    for labelx in label_mapping:
                        if label_mapping[labelx] == label:
                            userdata = self.getlabeluser(labelx)
                            if userdata is not None:
                                user = f"{userdata['firstName']} {userdata['lastName']}"
                                nameprefix = f"{userdata['firstName'][0]}.{userdata['lastName'][0]}"
                            label = labelx
                    # show label and confidence on video frame
                    prefix = "Unknown" if user is None else nameprefix # first and last latter of the name
                    text = f"{prefix}: {confidence:.2f}"
                    cv2.putText(video_frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

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