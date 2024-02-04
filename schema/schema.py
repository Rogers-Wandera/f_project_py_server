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