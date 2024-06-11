import numpy as np
from keras import layers
from keras.models import Sequential
import cv2 as cv
import os
import pathlib as pl
import keras
from conn.connector import Connection
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.layers import Dense, Flatten, Dropout, BatchNormalization
from keras.models import load_model

dbconnect = Connection()

class ImagePersonClassifier:
    """
    This class is for image classification and uses keras for training and testing
    The class uses oepncv for reading images and for other face operations like fine tunning the images
    """

    def __init__(self):
        self.labels = []
        self.accuracy = 0

    def _load_local_dataset(self, path, target_size=(224, 224), batch_size=32):
        """
        This function loads images from a dataset with data augmentation.
        """
        try:
            data_dir = pl.Path(path)
            datagen = ImageDataGenerator(
                 rescale=1./255,
                validation_split=0.2,
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest',
                brightness_range=[0.8,1.2],
            )

            train_dataset = datagen.flow_from_directory(
                data_dir,
                target_size=target_size,
                batch_size=batch_size,
                class_mode='sparse',
                subset='training'
            )

            val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

            val_dataset = val_datagen.flow_from_directory(
                data_dir,
                target_size=target_size,
                batch_size=batch_size,
                class_mode='sparse',
                subset='validation'
            )

            return train_dataset, val_dataset
        except Exception as e:
            raise e

    # def _load_local_dataset(self, path, target_size=(224, 224), batch_size=32):
    #     """
    #     This function loads images from a dataset like (mainfolder->foldername,foldername):
    #     - This functions loads the image and appends it to the images array which is later converted to numpy array
    #     - The function aswell saves the labels of the folder per images and also creates numerical labels \n

    #     Parameters:
    #     - path -> local images main folder path.
    #     - target_size -> tuple for the size of the image to be reshaped to default is (224,224)
    #     - batch_size -> batch size for the dataset \n

    #     Returns:
    #     - np.array: The processed images,labels,numericallabels.
    #     """
    #     try:
    #         data_dir = pl.Path(path)
            
    #         # training dataset
    #         train_dataset = keras.utils.image_dataset_from_directory(
    #             data_dir, validation_split=0.2, subset="training", seed=123,
    #             image_size=target_size, batch_size=batch_size)

    #         # validation dataset
    #         val_dataset = keras.utils.image_dataset_from_directory(
    #             data_dir, validation_split=0.2, subset="validation", seed=123,
    #             image_size=target_size, batch_size=batch_size
    #         )
    #         return train_dataset, val_dataset
    #     except Exception as e:
    #         raise e

    def _create_model_v1(self, num_classes, input_shape=(224, 224,3), activation="relu"):
        """
        This function creates a model using keras sequential model \n
        Parameters:
        - num_classes -> number of classes in the dataset.
        - input_shape -> tuple containing the shape of the input image \n

        Returns:
        - keras.Sequential: The created model
        """
        try:
            model = Sequential([
                layers.Rescaling(1./255, input_shape=input_shape),
                layers.Conv2D(16, 3, padding="same", activation="relu"),
                layers.MaxPool2D(),
                layers.Conv2D(32, 3, padding="same", activation="relu"),
                layers.MaxPool2D(),
                layers.Conv2D(64, 3, padding="same", activation="relu"),
                layers.MaxPool2D(),
                layers.Flatten(),
                layers.Dense(128, activation=activation),
                layers.Dense(num_classes)
            ])
            return model
        except Exception as e:
            raise e

    def _compile_model(self, model, optimizer="adam", loss="sparse", metrics="accuracy",
                       show_summary=True,learning_rate=None):
        """
        This function compiles the model with the given parameters \n
        Parameters:
        - model -> The model to be compiled
        - optimizer -> The optimizer to be used
        - loss -> The loss function to be used
        - metrics -> The metrics to be used
        - show_summary -> A flag to show or not the summary of the model \n
        Returns:
        - model -> The compiled model
        """
        try:
            if optimizer == "adam":
                if learning_rate is None:
                    optimizer = keras.optimizers.Adam()
                else:
                    optimizer = keras.optimizers.Adam(learning_rate)
            if loss == "sparse":
                loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            elif loss == "binary":
                loss = keras.losses.BinaryCrossentropy(from_logits=True)
            if metrics == "accuracy" and loss == "binary":
                metrics = [keras.metrics.BinaryAccuracy()]
            elif metrics == "accuracy" and loss == "sparse":
                metrics = [keras.metrics.CategoricalAccuracy()]
            model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
            if show_summary:
                model.summary()
            return model
        except Exception as e:
            raise e

    def _train_model(self, model, train_dataset, val_dataset, model_save_path, epochs=10):
        """
        This function trains the model \n
        Parameters:
        - model -> The compiled model to be trained
        - train_dataset -> The training dataset
        - val_dataset -> The validation dataset
        - model_save_path -> The path to save the model
        - epochs -> The number of epochs to be trained \n
        Returns:
        - model -> The trained model history
        """
        try:
            modal_path = os.path.join(os.getcwd(), "models", "models", f"{model_save_path}.keras")
            history = model.fit(
                train_dataset, validation_data=val_dataset, epochs=epochs)
            model.save(modal_path)
            return history
        except Exception as e:
            raise e
    
    def _dat_aug_model(self, model, train_dataset, val_dataset, model_save_path, epochs=10):
        """
        This function creates a model using keras functional API \n
        Parameters:
        - num_classes -> number of classes in the dataset.
        - input_shape -> tuple containing the shape of the input image \n
        Returns:
        - keras.Model: The created model
        """
        try:
            datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True)
            datagen.fit(train_dataset)
            modal_path = os.path.join(os.getcwd(), "models", "models", f"{model_save_path}.keras")
            history = model.fit(datagen.flow(train_dataset, batch_size=32),
                            validation_data=val_dataset, 
                            steps_per_epoch=len(train_dataset) / 32, 
                            epochs=epochs)
            model.save(modal_path)
            return history
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

    def _create_model_v2(self, num_classes, input_shape=(224, 224,3), activation="sigmoid"):
        """
        This function creates a model using keras sequential model with some regularization \n
        Parameters:
        - num_classes -> number of classes in the dataset.
        - input_shape -> tuple containing the shape of the input image \n

        Returns:
        - keras.Sequential: The created model
        """
        try:
            model = Sequential([
                layers.Rescaling(1./255, input_shape=input_shape),
                layers.Conv2D(16, 3, kernel_regularizer=keras.regularizers.l2(0.0001), padding="same", activation=activation),
                layers.MaxPool2D(),
                layers.Conv2D(32, 3, padding="same", activation=activation),
                layers.MaxPool2D(),
                layers.Conv2D(64, 3, padding="same", activation=activation),
                layers.MaxPool2D(),
                layers.Conv2D(128, 3, padding="same", activation=activation),
                layers.MaxPool2D(),
                layers.Flatten(),
                layers.Dense(256, kernel_regularizer=keras.regularizers.l2(0.0001), activation=activation),
                layers.Dropout(0.5),
                layers.Dense(num_classes, activation='softmax')
            ])
            return model
        except Exception as e:
            raise e
    
    def _create_model_v3(self, num_classes, input_shape=(224, 224,3), activation="relu"):
        """
        This function creates a model using keras sequential model with some regularization \n
        Parameters:
        - num_classes -> number of classes in the dataset.
        - input_shape -> tuple containing the shape of the input image \n

        Returns:
        - keras.Sequential: The created model
        """
        try:
            base_model = VGG16(include_top=False, input_shape=input_shape, weights='imagenet')
            base_model.trainable = False
            # model = Sequential([
            #     layers.Rescaling(1./255, input_shape=input_shape),
            #     layers.Conv2D(16, 3, kernel_regularizer=keras.regularizers.l2(0.0001), padding="same", activation=activation),
            #     layers.MaxPool2D(),
            #     layers.Conv2D(32, 3, padding="same", activation=activation),
            #     layers.MaxPool2D(),
            #     layers.Conv2D(64, 3, padding="same", activation=activation),
            #     layers.MaxPool2D(),
            #     layers.Conv2D(128, 3, padding="same", activation=activation),
            #     layers.MaxPool2D(),
            #     layers.Flatten(),
            #     layers.Dense(256, kernel_regularizer=keras.regularizers.l2(0.0001), activation=activation),
            #     layers.Dropout(0.7),
            #     layers.Dense(num_classes, activation='softmax')
            # ])
            model = Sequential([
                base_model, 
                Flatten(), 
                Dense(256, activation=activation),
                BatchNormalization(),
                Dropout(0.5),
                Dense(num_classes, activation='softmax')
            ])
            return model
        except Exception as e:
            raise e

    def _load_model(self, model_name):
        try:
            modalpath = os.path.join(
                os.getcwd(), "models", "models", f"{model_name}.keras")
            model = load_model(modalpath)
            return model
        except Exception as e:
            raise e

    def _predict_with_image_kr(self, image, model_name, input_shape=(224, 224, 3)):
        try:
            # check if the image is of the right shape
            if image.shape != input_shape:
                raise Exception(
                    "The shape of the image is not the same as the input shape")
            # normalise the image
            image = image / 255.0
            # predict the label of the image
            model = self._load_model(model_name)
            if model is None:
                raise Exception("Model not found")
            predictions = model.predict(np.array([image]))
            return predictions
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
            predicted_class_percentage = (np.exp(confidence) / np.sum(np.exp(top_4_confidences))) * 100
            
            predict_dict = {
                "predicted_class_index": predicted_class_index,
                "top_4_indices": top_4_indices,
                "top_4_confidences": top_4_confidences,
                "predictclass_confidence": confidence,
                "confidence_percentages": confidence_percentages,
                "predicted_class_percentage": predicted_class_percentage
            }
            return predict_dict
        except Exception as e:
            raise e

    def _show_predicted_people_kr(self, predict_dict, class_labels):
        try:
            predicted = []
            # check if predict_dict contains the required keys
            if not all(key in predict_dict for key in ["top_4_indices", "top_4_confidences"]):
                raise Exception(
                    "The predict_dict does not contain the required keys")
            # get the top 4 indices and confidences
            for i, (index, confidence) in enumerate(zip(predict_dict["top_4_indices"], predict_dict["confidence_percentages"]), 1):
                if isinstance(class_labels, dict):
                    label = list(class_labels.keys())[list(class_labels.values()).index(index)]
                else:
                    label = class_labels[index]
                user = dbconnect.findone("person", {"id": label, "isActive": 1})
                if user != None:
                    user_name = f"{user['firstName']} {user['lastName']}"
                    predicted.append({"rank": i+1, "label": user_name, "confidence": int(round(confidence)), "id": label})
            return predicted
        except Exception as e:
            raise e
    
    def _show_predicted_class(self, predictions, class_labels):
        try:
            predict = self._predicted_class(predictions)
            predicted_class = predict['predicted_class_index']
            # predicted_confidence = predict['predictclass_confidence']
            predicted_percentage = int(round(predict['predicted_class_percentage']))
            if isinstance(class_labels, dict):
                label = list(class_labels.keys())[list(class_labels.values()).index(predicted_class)]
            else:
                label = class_labels[predicted_class]
            user = dbconnect.findone("person", {"id": label, "isActive": 1})
            predicted = {}
            if user != None:
                user_name = f"{user['firstName']} {user['lastName']}"
                predicted = {"label": user_name, "confidence": predicted_percentage, "rank": 1, "id": label}
            return predicted
        except Exception as e:
            raise e
    
    def ModelFineTune(self, model_save_path,train_dataset, val_dataset,optimizer,loss,metrics,show_summary,
                      activation,num_classes, epochs=10):
        try:
            model = self._load_model(model_save_path)
            model.trainable = False
            model = Sequential([
                model, 
                Flatten(), 
                Dense(256, activation=activation),
                BatchNormalization(),
                Dropout(0.5),
                Dense(num_classes, activation='softmax')
            ])
            model = self._compile_model(model, optimizer=optimizer, loss=loss, metrics=metrics, show_summary=show_summary)
            history = self._train_model(model, train_dataset,val_dataset,model_save_path,epochs=epochs)
            return history
        except Exception as e:
            raise e
