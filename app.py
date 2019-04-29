import os
import cv2
import io
import numpy as np

from flask import Flask, request, render_template, send_from_directory

from core.detector import Detector
from core.recognizer import Recognizer

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
detector = Detector("model.ckpt-49491")
recog = Recognizer("shadownet.ckpt")

@app.route("/")
def index():
    return render_template("upload.html")


@app.route("/upload", methods=["POST"])
def upload():
    folder_name = 'images'
    '''
    # this is to verify that folder to upload to exists.
    if os.path.isdir(os.path.join(APP_ROOT, 'files/{}'.format(folder_name))):
        print("folder exist")
    '''
    target = os.path.join(APP_ROOT, 'static/{}'.format(folder_name))
    print(target)
    if not os.path.isdir(target):
        os.mkdir(target)
    print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        # This is to verify files are supported
        ext = os.path.splitext(filename)[1]
        if (ext == ".jpg") or (ext == ".png"):
            print("File supported moving on...")
        else:
            render_template("Error.html", message="Files uploaded are not supported...")
        destination = "/".join([target, filename])
        bio = io.BytesIO()
        upload.save(bio)
        new_img = detect(bio)
        cv2.imwrite(destination, new_img)

    # return send_from_directory("images", filename, as_attachment=True)
    return render_template("complete.html", filename=filename)

def detect(bio):
    img =  cv2.imdecode(np.frombuffer(bio.getvalue(), dtype='uint8'), 1)
    boxes = detector.detect(img)
    img_copy = img[:]
    for box in boxes:
        word_img = img[box[1]:box[3], box[0]:box[2]]
        word = recog.recognize(word_img)
        cv2.rectangle(img_copy, (box[0], box[1]), (box[2], box[3]), (125, 125, 0), 1)
        cv2.putText(img_copy, word, (box[0], box[1]), cv2.FONT_HERSHEY_PLAIN, 1, (25, 125, 0))

    return img_copy

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4555, debug=True)
