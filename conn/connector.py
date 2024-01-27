import mysql.connector as mysql
from conn.config import config

class Connection:
    def __init__(self):
        self.host = config["host"]
        self.user = config["user"]
        self.password = config["password"]
        self.database = config["database"]
        self.port = config["port"]
        self.connection = None
    
    def connect(self):
        try:
            self.connection = mysql.connect(
                host = self.host,
                user = self.user,
                password = self.password,
                database = self.database,
                port = self.port
            )
            if not self.connection.is_connected():
                raise mysql.Error("Connection not established")
            return self.connection
        except mysql.Error as err:
            raise err
    def executeQuery(self,query, values=None):
        try:
            if self.connection is None:
               self.connect()
            if not self.connection.is_connected():
               raise mysql.Error("Connection not established")
            
            cursor = self.connection.cursor(dictionary=True)
            if values:
               cursor.execute(query, values)
            else:
               cursor.execute(query)
            return cursor
        except mysql.Error as err:
            raise err
    
    def findall(self, table, conditions=None):
        try:
            query = "SELECT *FROM {}".format(table)
            cursor = self.executeQuery(query)
            results = cursor.fetchall()
            return results
        except mysql.Error as err:
            raise err
    
    def findone(self, table, conditions=None):
        try:
            query = "SELECT *FROM {}".format(table)
            if conditions is None:
                raise mysql.Error("No conditions provided")
            # check if conditions is not a dictionary
            if not isinstance(conditions, dict):
                raise mysql.Error("conditions must be a dictionary")
            values = None
            query += " WHERE " + " AND ".join([f"{key}=%s" for key in conditions])
            values = tuple(conditions.values())
            print(query)
            cursor = self.executeQuery(query, values)
            results = cursor.fetchone()
            return results
        except mysql.Error as err:
            raise err
    def disconnect(self):
        if self.connection is None:
            return
        if self.connection.is_connected():
            self.connection.close()
