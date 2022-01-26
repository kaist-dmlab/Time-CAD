from flask import Flask, flash, request, redirect, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename

from file_processor import preprocess_uploaded_file

app = Flask(__name__)
CORS(app, resources={r"/upload": {"origins": "http://localhost:3000"}})

ALLOWED_EXTENSIONS = {'csv', 'xls', 'xlsx'}


@app.route('/', methods=['GET'])
def index():
    return {
        'sender': 'Bot',
        'text': 'results[0]',
        'current_state': 'results[1]',
        'user_slot': 'results[2]'
    }


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = f'./uploaded_files/{filename}'
        file.save(filepath)
        return preprocess_uploaded_file(filepath)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port='5555', debug=True)
