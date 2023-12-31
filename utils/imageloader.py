import cloudinary
import cloudinary.api
import os
from collections import defaultdict
class ImageLoader:
    def __init__(self):
        self.cloudinaryconfig = {
            "cloud_name": os.getenv("CLOUD_NAME"),
            "api_key": os.getenv("CLOUD_API_KEY"),
            "api_secret": os.getenv("CLOUD_API_SECRET"),
            "secure": True
        }
        cloudinary.config(**self.cloudinaryconfig)
    
    def GetImageFolders(self, folder_path):
       try:
        results = cloudinary.api.resources(type='upload', prefix=folder_path, max_results=500)
        return results
       except Exception as e:
            raise e
    def OrganizePersonImages(self, folder_path):
       try:
           results = self.GetImageFolders(folder_path)
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
