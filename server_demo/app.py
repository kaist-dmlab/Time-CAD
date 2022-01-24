from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}


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
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('download_file', name=filename))
    return None

if __name__ == "__main__":
    app.run(host='0.0.0.0', port='5555', debug=True)
