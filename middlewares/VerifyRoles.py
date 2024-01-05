from conn import rolelist
from functools import wraps
from flask import jsonify,request
from utils.logger import Logger
security_logger = Logger()
USER_ROLES = rolelist.USER_ROLES

def verifyroles(*args):
   def decorator(request_function):
      @wraps(request_function)
      def roles_wrapper(*func_args, **kwargs):
        try:
            req = request
            if not hasattr(req, "roles") or not isinstance(req.roles, list):
                raise Exception("not authorized to access this route")
            roleList =  []
            for arg in args:
                roleList.append(arg)
            # check if the request role is in the roleList
            results = [role in roleList for role in req.roles]
            check_results = any(results)
            if not check_results:
                raise Exception("not authorized to access this route")
            
            user = request.user
            userinfo = f"User: {user['displayName']} id: {user['id']}"
            action = f"{req.method} {req.path}"
            security_logger.logaccess(userinfo, action)
        except Exception as e:
            return jsonify({'error': f'{e}'}), 401      
        return request_function(*func_args, **kwargs)
      return roles_wrapper
   return decorator

