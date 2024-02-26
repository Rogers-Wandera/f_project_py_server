audio_schema = {
    "type": "object",
    "properties":{
        "audio": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "minLength": 1},
                "filename": {"type": "string"},
                "mimetype": {"type": "string"},
                "size": {"type": "number"},
                "fieldname": {"type": "string"},
                "encoding": {"type": "string"},
                "destination": {"type": "string"},
            },
            "required": ["path"]
        }
    },
    "required": ["audio"]
}

train_schema = {
    "type": "object",
    "properties": {
        "version": {"type": "string", "enum": ["v1", "v2"]},
        "activation": {"type": "string", "enum": ["relu", "sigmoid"]},
        "remove": {"type": "number", "enum": [0, 1]},
        "download": {"type": "number", "enum": [0, 1]},
    }
}