from utils.main import MainAudioClassifier
import os
from keras.callbacks import ReduceLROnPlateau
import numpy as np

class PersonAudio(MainAudioClassifier):
    def __init__(self, label_path, model_path):
        super().__init__()
        self.mainmodalpath = "models"
        self.label_path = os.path.join(self.mainmodalpath, "labels", f"{label_path}.json")
        self.model_path = os.path.join(self.mainmodalpath, "models", f"{model_path}.keras")
        self.types = ["local", "cloudinary"]

        # create path if not exists
        if not os.path.exists(self.mainmodalpath):
            try:
                os.makedirs(self.mainmodalpath, exist_ok=True)
            except Exception as e:
               raise e

        full_model_path = os.path.join(self.mainmodalpath, "models")
        full_label_path = os.path.join(self.mainmodalpath, "labels")

        # create paths if not exits
        if not os.path.exists(full_model_path):
           try:
                os.makedirs(full_model_path, exist_ok=True)
           except Exception as e:
               raise e
        
        if not os.path.exists(full_label_path):
            try:
                os.makedirs(full_label_path, exist_ok=True)
            except Exception as e:
               raise e

    
    def _train_model(self, X_train,X_test, y_train, y_test, input_shape=(13, 174, 1)):
        try:
            X_train = X_train.reshape(X_train.shape[0], 13, 174, 1)
            X_test = X_test.reshape(X_test.shape[0], 13, 174, 1)

            label_length = len(self.labels)

            model = self._create_model(input_shape=input_shape, dense=label_length)

            model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
            callbacks = [reduce_lr]

            label_mappings = {user_id: idx for idx, user_id in enumerate(self.labels)}
            self._save_label_mappings(label_mappings, self.label_path)

            model.fit(X_train,y_train, validation_data=(X_test, y_test), epochs=10, batch_size=3, callbacks=callbacks)

            # save model
            return model.save(self.model_path)
        except Exception as e:
            raise e

    
    def _train_model_on_local_data(self, filepath,max_pad_len=174, test_size=0.2, random_state=42):
        try:
            X_train, X_test, y_train, y_test = self._load_local_dataset(path=filepath,max_pad_len=max_pad_len,test_size=test_size,random_state=random_state)
            self._train_model(X_train, X_test, y_train, y_test)
        except Exception as e:
            raise e
    
    def _train_model_on_cloudinary_data(self, path, max_pad_len=174, test_size=0.2, random_state=42):
        try:
            X_train,X_test,y_train,y_test = self._load_cloudinary_dataset(path,max_pad_len,test_size,random_state)
            self._train_model(X_train, X_test, y_train, y_test)
        except Exception as e:
            raise e
        
    
    def _train_emotions_model(self, path, type, max_pad_len=174, test_size=0.2, random_state=42):
        try:
            if type not in self.types:
                raise ValueError("Invalid type. Supported types: local, cloudinary")
            if type == "local":
                self._train_model_on_local_data(path, max_pad_len, test_size, random_state)
            elif type == "cloudinary":
                self._train_model_on_cloudinary_data(path, max_pad_len, test_size, random_state)
        except Exception as e:
            raise e
    
    def _train_person_audio_model(self, path, type, max_pad_len=174, test_size=0.2, random_state=42):
        try:
            if type not in self.types:
                raise ValueError("Invalid type. Supported types: local, cloudinary")
            if type == "local":
                self._train_model_on_local_data(path, max_pad_len, test_size, random_state)
            elif type == "cloudinary":
                self._train_model_on_cloudinary_data(path, max_pad_len, test_size, random_state)
        except Exception as e:
            raise e
    
    def _predict_person_audio(self, path, confidence_threshold=0.7):
        try:
            model = self._load_model(self.model_path)
            mfccs = self._extract_mfcc(path)
            mfccs = mfccs.reshape(1, 13, 174, 1)
            predictions = model.predict(mfccs)
            predicted_class_index = np.argmax(predictions)
            confidence = predictions[0, predicted_class_index]

            # Load label mappings
            label_mappings = self._load_label_mappings(self.label_path)

            # Check if the confidence is below the threshold or if the label is unknown
            if confidence < confidence_threshold:
                return predictions, None, "unknown", confidence, round(confidence * 100, 1)

            # Map predicted index to label
            predicted_label = "unknown"
            for label in label_mappings:
                if label_mappings[label] == predicted_class_index:
                    predicted_label = label
                    break

            percentage_confidence = round(confidence * 100, 1)
            return predictions, predicted_class_index, predicted_label, confidence, percentage_confidence

        except Exception as e:
            raise e

    
