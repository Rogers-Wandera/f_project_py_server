from utils.main import MainAudioClassifier
import os
from keras.callbacks import ReduceLROnPlateau
from conn.connector import Connection
import numpy as np
import math
import matplotlib.pyplot as plt
connect = Connection()
class PersonAudio(MainAudioClassifier):
    def __init__(self, label_path, model_path):
        super().__init__()
        self.mainmodalpath = "models"
        self.label_path = os.path.join(self.mainmodalpath, "labels", f"{label_path}.json")
        self.model_path = os.path.join(self.mainmodalpath, "models", f"{model_path}.keras")
        self.types = ["local", "cloudinary"]
        self.trainedItemsCount = 0

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

            results = model.fit(X_train,y_train, validation_data=(X_test, y_test), epochs=10, batch_size=3, callbacks=callbacks)
            self.trainedItemsCount = len(X_train)
            # save model
            return model.save(self.model_path), results
        except Exception as e:
            raise e

    
    def _train_model_on_local_data(self, filepath,max_pad_len=174, test_size=0.2, random_state=42):
        try:
            X_train, X_test, y_train, y_test = self._load_local_dataset(path=filepath,max_pad_len=max_pad_len,test_size=test_size,random_state=random_state)
            data = self._train_model(X_train, X_test, y_train, y_test)
            return data[1]
        except Exception as e:
            raise e
    
    def _train_model_on_cloudinary_data(self, path, max_pad_len=174, test_size=0.2, random_state=42):
        try:
            X_train,X_test,y_train,y_test = self._load_cloudinary_dataset(path,max_pad_len,test_size,random_state)
            data =  self._train_model(X_train, X_test, y_train, y_test)
            return data[1]
        except Exception as e:
            raise e
        
    
    def _train_emotions_model(self, path, type, max_pad_len=174, test_size=0.2, random_state=42):
        try:
            if type not in self.types:
                raise ValueError("Invalid type. Supported types: local, cloudinary")
            if type == "local":
                return self._train_model_on_local_data(path, max_pad_len, test_size, random_state)
            elif type == "cloudinary":
                return self._train_model_on_cloudinary_data(path, max_pad_len, test_size, random_state)
        except Exception as e:
            raise e
    
    def _train_person_audio_model(self, path, type, max_pad_len=174, test_size=0.2, random_state=42):
        try:
            if type not in self.types:
                raise ValueError("Invalid type. Supported types: local, cloudinary")
            if type == "local":
                return self._train_model_on_local_data(path, max_pad_len, test_size, random_state)
            elif type == "cloudinary":
                return self._train_model_on_cloudinary_data(path, max_pad_len, test_size, random_state)
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

            percentage_confidence = math.floor(confidence * 100)
            return predictions, predicted_class_index, predicted_label, confidence, percentage_confidence

        except Exception as e:
            raise e
        
    def _show_predicted_person(self, path,confidence_threshold=0.7):
        try:
            predictions, predicted_class_index, predicted_label, confidence, percentage_confidence = self._predict_person_audio(path, confidence_threshold)
            predicted = self._predicted_class(predictions)
            top_indices = predicted['top_4_indices']
            top_labels = predicted['top_4_indices_labels']
            print(top_labels)
            predictedPerson = {}
            otherpredictions = []
            if predicted_label != "unknown":
               user = connect.findone("person", {"id": predicted_label, "isActive": 1})
               if user != None:
                   user_name = f"{user['firstName']} {user['lastName']}"
                   predictedPerson = {"label": user_name, "confidence": percentage_confidence, "id": predicted_label}
            
            if len(top_indices) > 0:
               for label, con, perc in top_labels:
                   if label != predicted_label:
                       user = connect.findone("person", {"id": label, "isActive": 1})
                       if user != None:
                           user_name = f"{user['firstName']} {user['lastName']}"
                        #    percentg_conf = round(confidence * 100, 1)
                           otherpredictions.append({"label": user_name, "confidence": perc, "id": label})
            return predictedPerson,otherpredictions
        except Exception as e:
            raise e
    
    def _predicted_class(self, predictions):
        try:
            predicted_class_index = np.argmax(predictions)
            top_4_indices = np.argsort(predictions[0])[::-1][:4]
            
            # Remove the predicted_class_index from top_4_indices
            top_4_indices = top_4_indices[top_4_indices != predicted_class_index]
            
            top_4_confidences = predictions[0, top_4_indices]
            confidence = predictions[0, predicted_class_index]
            confidence_percentages = (np.exp(top_4_confidences) /
                                    np.sum(np.exp(top_4_confidences))) * 100
            
            # Calculate percentage of predicted_class_index
            predicted_class_percentage = math.floor(confidence * 100)
            label_mappings = self._load_label_mappings(self.label_path)

            top_4_indices_labels = []
            for index in top_4_indices:
                for label in label_mappings:
                    if label_mappings[label] == index:
                        other_confidence = predictions[0, index]
                        confidence_percentage = math.floor((np.exp(other_confidence) / np.sum(np.exp(top_4_confidences))) * 100)
                        top_4_indices_labels.append((label, other_confidence,confidence_percentage))
                        break
            
            predict_dict = {
                "predicted_class_index": predicted_class_index,
                "top_4_indices": top_4_indices,
                "top_4_confidences": top_4_confidences,
                "predictclass_confidence": confidence,
                "confidence_percentages": confidence_percentages,
                "predicted_class_percentage": predicted_class_percentage,
                "top_4_indices_labels": top_4_indices_labels
            }
            return predict_dict
        except Exception as e:
            raise e
        

    def _evaluate_model(self, history):
        """
        This function evaluates the model and returns the evaluation \n
        Parameters:
        - history -> The history of the model \n
        Returns:
        - evaluation_dict -> The evaluation of the model containing train and test accuracy and their losses
        """
        try:
            # train accuracy and loss
            train_accuracy = history.history['accuracy']
            loss = history.history['loss']
            # test accuracy and loss
            test_accuracy = history.history['val_accuracy']
            val_loss = history.history['val_loss']
            evaluation_dict = {
                "train_evaluation": (train_accuracy, loss),
                "test_evaluation": (test_accuracy, val_loss)
            }
            return evaluation_dict
        except Exception as e:
            raise e
        
    def _display_evaluation(self,history):
        try:
           # accuracy
            eval_dict = self._evaluate_model(history)
            train_accuracy = eval_dict["train_evaluation"][0]
            test_accuracy =  eval_dict["test_evaluation"][0]

            # loss
            loss = eval_dict["train_evaluation"][1]
            val_loss = eval_dict["test_evaluation"][1]

            print(f"Test Accuracy: {test_accuracy[-1] * 100:.2f}%")
            print(f"Loss: {val_loss[-1]}")

            print(f"Train Accuracy: {train_accuracy[-1] * 100:.2f}%")
            print(f"Loss: {loss[-1]}")
            return {
                "Test": [f"Test Accuracy: {test_accuracy[-1] * 100:.2f}%", f"Loss: {val_loss[-1]}"],
                "Train": [f"Train Accuracy: {train_accuracy[-1] * 100:.2f}%", f"Loss: {loss[-1]}"]
            }
        except Exception as e:
            raise e

    
