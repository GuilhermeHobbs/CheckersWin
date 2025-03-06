from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # This enables CORS for all routes

print("Uéeee")

@app.route('/move')
def ask_name():
    print("aloooo")
    #logits, _ = m(context.int())
    #print(logits)

    print("Eh assim")
    
    return "hello"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)  # Render requires explicit host/port
