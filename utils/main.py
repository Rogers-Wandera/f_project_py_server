from sklearn.model_selection import train_test_split
from keras.models import Sequential,load_model
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from keras.callbacks import ReduceLROnPlateau
import numpy as np
import librosa
import os
import io
import json
from utils.imageloader import ImageLoader
import requests


class MainAudioClassifier(ImageLoader):
    def __init__(self):
        super().__init__()
        self.labels = []
    
    def _save_label_mappings(self, label_mappings, file_path):
        try:
            with open(file_path, 'w') as json_file:
                json.dump(label_mappings, json_file)
        except Exception as e:
            raise e
    
    def _load_label_mappings(self, file_path):
        try:
            with open(file_path, "r") as json_file:
                label_mapping = json.load(json_file)
            return label_mapping
        except Exception as e:
            raise e
    
    def _extract_mfcc(self, file_path, max_pad_len=174):
        try:

            if file_path.startswith(('http://', 'https://')):
                response = requests.get(file_path)
                y, sr = librosa.load(io.BytesIO(response.content), mono=True, sr=None)
            else:
                y,sr = librosa.load(file_path, mono=True,sr=None)
            
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

            if mfccs.shape[1] < max_pad_len:
                pad_width = max_pad_len - mfccs.shape[1]
                mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
            else:
                mfccs = mfccs[:, :max_pad_len]

            mfccs.flatten()
            return mfccs   
        except Exception as e:
            raise e
    
    def _load_local_dataset(self, path, max_pad_len=174, test_size=0.2, random_state=42):
       try:
            mainpath = os.listdir(path)
            x,y = [], []

            for i, emotion in enumerate(mainpath):
                emotion_path = os.path.join(path, emotion)
                self.labels.append(emotion)
                for filename in os.listdir(emotion_path):
                    file_path = os.path.join(emotion_path, filename)
                    mfccs = self._extract_mfcc(file_path, max_pad_len)
                    x.append(mfccs)
                    y.append(i)
            
            x = np.array(x)
            y = np.array(y)
            X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=test_size, random_state=random_state)
            return X_train, X_test, y_train, y_test
       except Exception as e:
            raise e
    
    def _load_cloudinary_dataset(self, path, max_pad_len=174, test_size=0.2, random_state=42, max_results=500):
        try:
            x = []
            y = []
            results = self.OrganizePersonAudios(folder_path=path, max_results=max_results)
            if results is None:
                raise Exception("No Audio files found")
            for i, folder in enumerate(results):
                self.labels.append(folder)
                for file in results[folder]:
                    audio_path = file['secure_url']
                    mfccs = self._extract_mfcc(audio_path, max_pad_len)
                    x.append(mfccs)
                    y.append(i)
                
            x = np.array(x)
            y = np.array(y)
            X_train,X_test,y_train,y_test = train_test_split(x,y, test_size=test_size, random_state=random_state)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            raise e
            
    def _create_model(self, input_shape,dense=7):
        try:
            model = Sequential()
            model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
            model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))
            model.add(Flatten())
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(dense, activation='softmax'))
            model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            return model
        except Exception as e:
            raise e
    
    def _load_model(self, model_name_path):
        try:
            model = load_model(model_name_path)
            return model
        except Exception as e:
            raise e
        
    