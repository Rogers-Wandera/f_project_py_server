from flask import request, jsonify
from schema.schema import audio_schema
from jsonschema import validate,ValidationError
from utils.personaudio import PersonAudio

audio = PersonAudio(label_path="personsaudiolabels",
                  model_path="personsaudiomodel")

def TrainAudioClassifier():
    try:
        results = audio._train_person_audio_model("personsaudio", type="cloudinary")
        evaluation = audio._display_evaluation(history=results)
        return jsonify({"msg": "Audio model trained successfully", 
                        "evaluation": evaluation, 
                        "itemsCount": audio.trainedItemsCount,"modelName": "personsaudiomodel"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

def PredictAudio():
    try:
        validate(schema=audio_schema, instance=request.json)
        (predicted, others) = audio._show_predicted_person(request.json['audio']['path'])
        prediction = {"predict": predicted, "others": others}
        return jsonify(prediction), 200
    except ValidationError as ve:
        return jsonify({"error":ve.message}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
