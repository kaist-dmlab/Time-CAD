from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def index():
    return {
        'sender': 'Bot',
        'text': 'results[0]',
        'current_state': 'results[1]',
        'user_slot': 'results[2]'
    }


@app.route('/upload', methods=['POST'])
def upload_file():
    # get uploaded file and return its .json version for displaying charts
    return None

if __name__ == "__main__":
    app.run(host='0.0.0.0', port='5555', debug=True)