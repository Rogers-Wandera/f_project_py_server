import mysql.connector as mysql

class Connection:
    def __init__(self, host,user,password,database,port):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.port = port
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
            if self.connection.is_connected():
                return self.connection
        except mysql.connector.Error as err:
            raise err
    def execute_query(self, query):
        try:
            cursor = self.connection.cursor()
            cursor.execute(query)
            results = cursor.fetchall()
            cursor.close()
            return results
        except mysql.connector.Error as err:
            print(f"Error: {err}")
            return None
    def disconnect(self):
        if self.connection.is_connected():
            self.connection.close()
