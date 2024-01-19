from sklearn.model_selection import train_test_split
from keras.models import Sequential,load_model
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from keras.callbacks import ReduceLROnPlateau
import numpy as np
import librosa
import os
import json

from utils.imageloader import ImageLoader

class AudioClassifier:
    def __init__(self):
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
    
    def _extract_mfcc(self, file_path, max_pad_len=174):
        try:
            y,sr = librosa.load(file_path, mono=True,sr=None)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

            if mfccs.shape[1] < max_pad_len:
                pad_width = max_pad_len - mfccs.shape[1]
                mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
            else:
                mfccs = mfccs[:, :max_pad_len]

            return mfccs   
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
    
    def _train_on_local_data(self, file_path, label_path='audio_label_mapping.json'):
        try:
            X_train, X_test, y_train, y_test = self._load_local_dataset(path=file_path)
            X_train = X_train.reshape(X_train.shape[0], 13, 174, 1)
            X_test = X_test.reshape(X_test.shape[0], 13, 174, 1)
            label_length = len(self.labels)
            model = self._create_model(input_shape=(13, 174, 1), dense=label_length)
            # compile model
            model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
            callbacks = [reduce_lr]

            # save label mappings
            label_mappings = {user_id: idx for idx, user_id in enumerate(self.labels)}
            self._save_label_mappings(label_mappings, label_path)

            # train model
            model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=3, callbacks=callbacks)

            # save the model
            return model.save('audioclassifiermodel.keras')
        except Exception as e:
            raise e
    
    def _load_model(self, model_path):
        try:
            model = load_model(model_path)
            return model
        except Exception as e:
            raise e
    
    def _predict_audio(self, model_path, file_path, label_path='audio_label_mapping.json'):
        try:
            model = self._load_model(model_path)
            mfccs = self._extract_mfcc(file_path)
            mfccs = mfccs.reshape(1, 13, 174, 1)
            predictions = model.predict(mfccs)
            predicted_class_index = np.argmax(predictions)
            label_mappings = self._load_label_mappings(label_path)
            predicted_label = 'Unknown'
            for label in label_mappings:
                if label_mappings[label] == predicted_class_index:
                    predicted_label = label
                    break
            confidence = predictions[0, predicted_class_index]
            percentage_confidence = round(confidence * 100, 1)
            return predictions, predicted_class_index, predicted_label, confidence, percentage_confidence
        except Exception as e:
            raise e
    


obj = AudioClassifier()
dataset = r"D:\user\sockets_web\emotions"
audiopath = r"D:\user\sockets_web\emotions\sadness\s20 (6).wav"


# obj._train_on_local_data(file_path=dataset)
prediction = obj._predict_audio(model_path="audioclassifiermodel.keras", file_path=audiopath)
print(prediction)


        


# # function to extract mfcc (Mel-Frequency Cepstral Coefficients (MFCCs)) features
# # from the audio file

# def extract_mfcc(file_path, max_pad_len=174):
#     wave,sr = librosa.load(file_path, mono=True,sr=None)
#     mfccs = librosa.feature.mfcc(y=wave, sr=sr, n_mfcc=13)

#     if mfccs.shape[1] < max_pad_len:
#         pad_width = max_pad_len - mfccs.shape[1]
#         mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
#     else:
#         mfccs = mfccs[:, :max_pad_len]

#     return mfccs

# def load_and_preprocess_dataset(path, max_pad_len=174, test_size=0.2, random_state=42):
    # emotions = os.listdir(path)
    # x,y = [], []
    # for i, emotion in enumerate(emotions):
    #     emotion_path = os.path.join(path, emotion)
    #     for filename in os.listdir(emotion_path):
    #         file_path = os.path.join(emotion_path, filename)
    #         mfccs = extract_mfcc(file_path, max_pad_len)
    #         x.append(mfccs)
    #         y.append(i)
    
    # x = np.array(x)
    # y = np.array(y)

    # X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=test_size, random_state=random_state)
    # return X_train, X_test, y_train, y_test

# dataset = r"D:\user\sockets_web\emotions"
# X_train, X_test, y_train, y_test = load_and_preprocess_dataset(dataset, test_size=0.2, random_state=42)

# X_train = X_train.reshape(X_train.shape[0], 13, 174, 1)
# X_test = X_test.reshape(X_test.shape[0], 13, 174, 1)

# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(13, 174, 1)))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Flatten())
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(5, activation='softmax'))

# # compile model
# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
# callbacks = [reduce_lr]

# # train model
# model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=3, callbacks=callbacks)


# audiopath = r"D:\user\sockets_web\emotions\sadness\s20 (6).wav"
# new_mfccs = extract_mfcc(audiopath)

# new_mfccs = new_mfccs.reshape(1, 13, 174, 1)
# predictions = model.predict(new_mfccs)
# print(predictions)
# emotion_labels = ["Fear", "Happy", "Disgust", "Anger", "Sadness"]

# for emotion, confidence in zip(emotion_labels, predictions[0]):
#     print(f"{emotion}: {confidence * 100:.2f}%")
# predicted_emotion = emotion_labels[np.argmax(predictions)]

# print("Predicted Emotion:", predicted_emotion)