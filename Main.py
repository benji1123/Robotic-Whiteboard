import os
from flask import Flask, flash, request, redirect, url_for, render_template                                           
import threading
import polar
import queue

#somewhere accessible to both:
thread_queue = queue.Queue()

app = Flask(__name__, static_url_path='/static')

f = __file__

@app.route('/')
def home():
    return render_template('home.html');

@app.route('/upload', methods=['POST'])
def upload_file():
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
    file = request.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file:
        filename = file.filename
        full_path = os.path.join(os.path.dirname(os.path.abspath(f)), 'uploads', filename)
        file.save(full_path)
        thread_queue.put(full_path)
        return 'File successfully uploaded'
    return 'There was an error'

def flaskThread():
    app.run(debug=True, use_reloader=False)

if __name__ == "__main__":
    threading.Thread(target=flaskThread).start()
    while True:
        try:
            path = thread_queue.get() #doesn't block
            print("starting job " + path)
            polar.run(path)
        except:
            print("Exception parsing image")
    
