from flask import Flask, jsonify, request, redirect
import makeup
import PIL
import base64
import io

app = Flask(__name__)

def loadImageData(bytes, mode='RGB'):
    bytesIO = io.BytesIO(bytes)
    imageLoaded = face_recognition.load_image_file(bytesIO)
    return imageLoaded

def imageToBytes(image):
    bytesIO = io.BytesIO()
    PIL.Image.fromarray(image).save(bytesIO)
    bytesIO.seek(io.SEEK_SET)
    data = bytesIO.read(-1)
    bytesIO.close()
    return data



@app.route('/', methods=['POST'])
def makeup_image():
    imageBase64 = request.form['image']
    r,g,b,a = request.form['r'], request.form['g'], request.form['b'],request.form['a']
    if not imageBase64 or not r or not g or not b or not a:
        result = {
            "errno": 1
        }
        return jsonify(result)
    

    imageData = base64.standard_b64decode(imageBase64)
    loadedImage = loadImageData(imageData)
    finishedImage = makeup.putMakeupOn(loadedImage,r,g,b,a)
    finishedImageReturn = base64.standard_b64encode(imageToBytes(finishedImage))
    result = {
        "errno": 0,
        "data": finishedImageReturn
    }
    return jsonify(result)


