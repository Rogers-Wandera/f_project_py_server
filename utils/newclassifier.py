import numpy as np
import tensorflow as tf
from keras import layers
from keras.models import Sequential
import cv2 as cv
import os
import pathlib as pl
import keras


class ImagePersonClassifier:
    """
    This class is for image classification and uses keras for training and testing
    The class uses oepncv for reading images and for other face operations like fine tunning the images
    """

    def __init__(self):
        self.labels = []
        self.accuracy = 0

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
            modeFile = "models/models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
            configFile = "models/models/deploy.prototxt"
            net = cv.dnn.readNetFromCaffe(modeFile, configFile)
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

    def _load_local_dataset(self, path, target_size=(224, 224), batch_size=32):
        """
        This function loads images from a dataset like (mainfolder->foldername,foldername):
        - This functions loads the image and appends it to the images array which is later converted to numpy array
        - The function aswell saves the labels of the folder per images and also creates numerical labels \n

        Parameters:
        - path -> local images main folder path.
        - target_size -> tuple for the size of the image to be reshaped to default is (224,224)
        - batch_size -> batch size for the dataset \n

        Returns:
        - np.array: The processed images,labels,numericallabels.
        """
        try:
            data_dir = pl.Path(path)
            # training dataset
            train_dataset = keras.utils.image_dataset_from_directory(
                data_dir, validation_split=0.2, subset="training", seed=123,
                image_size=target_size, batch_size=batch_size)

            # validation dataset
            val_dataset = keras.utils.image_dataset_from_directory(
                data_dir, validation_split=0.2, subset="validation", seed=123,
                image_size=target_size, batch_size=batch_size
            )
            return train_dataset, val_dataset
        except Exception as e:
            raise e

    def _load_from_dataset(self, dataset, target_size=(224, 224), batch_size=32):
        """
        This function loads images from a dataset like (mainfolder->foldername,foldername):
        - This functions loads the image and appends it to the images array which is later converted to numpy array
        - The function aswell saves the labels of the folder per images and also creates numerical labels \n

        Parameters:
        - url -> url of the dataset
        - target_size -> tuple for the size of the image to be reshaped to default is (224,224)
        - batch_size -> batch size for the dataset \n

        Returns:
        - np.array: The processed images,labels,numericallabels.
        """
        try:
            train_dataset = keras.utils.image_dataset_from_directory(
                dataset, image_size=target_size, batch_size=batch_size
            )
            val_dataset = keras.utils.image_dataset_from_directory(
                dataset, image_size=target_size, batch_size=batch_size
            )
            return train_dataset, val_dataset
        except Exception as e:
            raise e

    def _create_model_v1(self, num_classes, input_shape=(224, 224), activation="relu"):
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

    def _compile_model(self, model, optimizer="adam", loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"],
                       show_summary=True):
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
            history = model.fit(
                train_dataset, validation_data=val_dataset, epochs=epochs)
            model.save(f"{model_save_path}.keras")
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

    def _create_model_v2(self, num_classes, input_shape=(224, 224), activation="sigmoid"):
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
                layers.Conv2D(16, 3, kernel_regularizer=keras.regularizers.l2(
                    0.0001), padding="same", activation="relu"),
                layers.MaxPool2D(),
                layers.Conv2D(32, 3, padding="same", activation="relu"),
                layers.MaxPool2D(),
                layers.Conv2D(64, 3, padding="same", activation="relu"),
                layers.MaxPool2D(),
                layers.Flatten(),
                layers.Dense(128, kernel_regularizer=keras.regularizers.l2(
                    0.0001), activation=activation),
                layers.Dropout(0.5),
                layers.Dense(num_classes)
            ])
            return model
        except Exception as e:
            raise e

    def _load_model(self, model_name):
        try:
            modalpath = os.path.join(
                os.getcwd(), "models", "models", f"{model_name}.keras")
            model = keras.models.load_model(modalpath)
            return model
        except Exception as e:
            raise e

    def _predict_with_image(self, image, model_name, input_shape=(224, 224, 3)):
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
            top_4_confidences = predictions[0, top_4_indices]
            confidence = predictions[0, predicted_class_index]
            confidence_percentages = (np.exp(top_4_confidences) /
                                      np.sum(np.exp(top_4_confidences))) * 100
            predict_dict = {
                "predicted_class_index": predicted_class_index,
                "top_4_indices": top_4_indices,
                "top_4_confidences": top_4_confidences,
                "predictclass_confidence": confidence,
                "confidence_percentages": confidence_percentages
            }
            return predict_dict
        except Exception as e:
            raise e

    def _show_predicted_people(self, predict_dict, class_labels):
        try:
            predicted = []
            # check if predict_dict contains the required keys
            if not all(key in predict_dict for key in ["top_4_indices", "top_4_confidences"]):
                raise Exception(
                    "The predict_dict does not contain the required keys")
            # get the top 4 indices and confidences
            for i, (index, confidence) in enumerate(zip(predict_dict["top_4_indices"], predict_dict["confidence_percentages"]), 1):
                label = class_labels[index]
                predicted.append(
                    {"rank": i+1, "label": label, "confidence": confidence}
                )
            return predicted
        except Exception as e:
            raise e
