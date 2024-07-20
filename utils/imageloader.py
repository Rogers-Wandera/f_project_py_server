import cloudinary
import cloudinary.api
import os
from io import BytesIO
from collections import defaultdict
import requests
from PIL import Image
import shutil
import cv2 as cv
class ImageLoader:
    def __init__(self):
        self.cloudinaryconfig = {
            "cloud_name": os.getenv("CLOUD_NAME"),
            "api_key": os.getenv("CLOUD_API_KEY"),
            "api_secret": os.getenv("CLOUD_API_SECRET"),
            "secure": True
        }
        cloudinary.config(**self.cloudinaryconfig)
        modeFile = os.path.join(os.getcwd(), "models", "models", "cafe.caffemodel")
        configFile = os.path.join(os.getcwd(), "models", "models", "deploy.prototxt")
        self.net = cv.dnn.readNetFromCaffe(configFile, modeFile)
        if self.net is None:
            raise Exception("Error loading network")

    
    def _face_detection(self, image, conf_threshold=0.8):
        """
        This function is for face detection using OpenCv DNN
            - This function takes an image and uses ResNet-10 Architecture as the backbone
            - This function uses Floating point 16 version of the original caffe implementation \n
        Parameters:
        - image (array of the read image): The image read from the either local or online source. \n

        Returns:
        - np.array: The processed image.
        """
        try:
            modeFile = os.path.join(os.getcwd(), "models", "models", "cafe.caffemodel")
            configFile = os.path.join(os.getcwd(), "models", "models", "deploy.prototxt")
            net = cv.dnn.readNetFromCaffe(configFile, modeFile)
            if net is None:
                raise Exception("Error loading network")
            blob = cv.dnn.blobFromImage(image=image, scalefactor=1.0, size=(
                300, 300), mean=(104.0, 177.0, 123.0), swapRB=False, crop=False)
            net.setInput(blob)
            detections = net.forward()
            bbox = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > conf_threshold:
                    x1 = int(detections[0, 0, i, 3] * image.shape[1])
                    y1 = int(detections[0, 0, i, 4] * image.shape[0])
                    x2 = int(detections[0, 0, i, 5] * image.shape[1])
                    y2 = int(detections[0, 0, i, 6] * image.shape[0])
                    face = image[y1:y2, x1:x2]
                    bbox.append(face)
            return bbox
        except Exception as e:
            raise e
    
    def _bounding_boxes(self, image, conf_threshold=0.8):
        """
        This function is for face detection using OpenCv DNN
            - This function takes an image and uses ResNet-10 Architecture as the backbone
            - This function uses Floating point 16 version of the original caffe implementation \n
        Parameters:
        - image (array of the read image): The image read from the either local or online source. \n

        Returns:
        - np.array: The processed image.
        """
        try:
            # modeFile = os.path.join(os.getcwd(), "models", "models", "cafe.caffemodel")
            # configFile = os.path.join(os.getcwd(), "models", "models", "deploy.prototxt")
            # net = cv.dnn.readNetFromCaffe(configFile, modeFile)
            if self.net is None:
                raise Exception("Error loading network")
            blob = cv.dnn.blobFromImage(image=image, scalefactor=1.0, size=(
                300, 300), mean=(104.0, 177.0, 123.0), swapRB=False, crop=False)
            self.net.setInput(blob)
            detections = self.net.forward()
            bbox = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > conf_threshold:
                    x1 = int(detections[0, 0, i, 3] * image.shape[1])
                    y1 = int(detections[0, 0, i, 4] * image.shape[0])
                    x2 = int(detections[0, 0, i, 5] * image.shape[1])
                    y2 = int(detections[0, 0, i, 6] * image.shape[0])
                    bbox.append((x1, y1, x2, y2))
            return bbox
        except Exception as e:
            raise e
    
    def GetImageFolders(self, folder_path, type='upload', max_results=500):
       try:
        results = cloudinary.api.resources(type=type, prefix=folder_path, max_results=max_results)
        return results
       except Exception as e:
            raise e
    def OrganizePersonImages(self, folder_path,type='upload', max_results=500):
       try:
           results = self.GetImageFolders(folder_path,type=type, max_results=max_results)
           resources = results['resources']
           user_images = defaultdict(list)
           for resource in resources:
               folder_parts = resource['folder'].split("/")
               userid = folder_parts[-1]
               user_images[userid].append({
                  "public_id": resource["public_id"],
                  "secure_url": resource["secure_url"],
                  "label": userid
               })
           return user_images
       except Exception as e:
            raise e
    
    def OrganizePersonAudios(self, folder_path, max_results=500):
        try:
            results = cloudinary.api.resources(type="upload", resource_type="raw",max_results=max_results, prefix=folder_path)
            resources = results['resources']
            if len(resources) <= 0:
                return None
            user_audios = defaultdict(list)
            for resource in resources:
               folder_parts = resource['folder'].split("/")
               userid = folder_parts[-1]
               user_audios[userid].append({
                  "public_id": resource["public_id"],
                  "secure_url": resource["secure_url"],
                  "label": userid
               })
            return user_audios
        except Exception as e:
            raise e
    
    def _get_read_images(self, folder_path, type='upload', max_results=500, save_folder="persons"):
        try:
           user_data = self.OrganizePersonImages(folder_path,type=type, max_results=max_results)
           downloaded_resources = 0
           save_folder_path = os.path.join(os.getcwd(), save_folder)
           os.makedirs(save_folder_path, exist_ok=True)
           #check if user_data is empty
           if len(user_data) <= 0:
               raise Exception("User data is empty")
           
           for data in user_data:
               user_subfolders = os.path.join(save_folder_path, data)
               os.makedirs(user_subfolders, exist_ok=True)
               for index ,image_data in enumerate(user_data[data]):
                   image_url = image_data["secure_url"]
                   response = requests.get(image_url)
                   image = Image.open(BytesIO(response.content))

                   uniquename = f"{data}_{index}.{image.format.lower()}"
                   save_path = os.path.join(user_subfolders, uniquename)
                   image.save(save_path)
                   downloaded_resources += 1
           return downloaded_resources
        except Exception as e:
            raise e
    
    def _remove_cloud_folder(self, save_folder="persons"):
        try:
            save_folder_path = os.path.join(os.getcwd(), save_folder)
            if os.path.exists(save_folder_path):
                shutil.rmtree(save_folder_path)
                return True
            else:
                return False
        except Exception as e:
            raise e