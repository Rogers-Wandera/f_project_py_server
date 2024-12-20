from flask import jsonify, request
from flask_jwt_extended import decode_token, exceptions
import time
from functools import wraps

def verifyjwt(request_function):
    @wraps(request_function)
    def jwt_verify_wrapper(*args, **kwargs):
        try:
            auth_header = request.headers.get('Authorization', '')
            if not auth_header.startswith("Bearer"):
                return jsonify({"msg": "You are not authorized to access this route"}), 401
            Token = auth_header.split(" ")[1]
            decoded = decode_token(encoded_token=Token,allow_expired=True)
            if "exp" in decoded and decoded['exp'] < time.time():
                raise exceptions.JWTDecodeError("Token has expired")
            user = decoded.get("user")
            request.user = {
                "id": user['id'],
                "displayName": user['displayName'],
            }
            request.roles = user['roles']
        except exceptions.JWTDecodeError as e:
            return jsonify({'error': f'{e}'}), 403
        except exceptions.JWTExtendedException as e:
            return jsonify({'error': f'{e}'}), 403

        return request_function(*args, **kwargs)

    return jwt_verify_wrapper
