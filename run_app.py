from flask import Flask, jsonify, request, redirect
import face_recognition
import makeup
import PIL
import base64
import io
import numpy as np

app = Flask(__name__)

def loadImageFile(file, ext='', mode='RGB'):
    im = PIL.Image.open(file,formats=['JPEG','PNG','BITMAP'])
    if mode:
        im = im.convert(mode)
    return np.array(im)

def loadImageData(data, mode='RGB'):
    bytesIO = io.BytesIO(data)
    bytesIO.seek(io.SEEK_SET)
    imageLoaded = loadImageFile(bytesIO)
    return imageLoaded

def imageToBytes(image):
    bytesIO = io.BytesIO()
    PIL.Image.fromarray(image).save(bytesIO,'JPEG')
    bytesIO.seek(io.SEEK_SET)
    data = bytesIO.read(-1)
    bytesIO.close()
    return data



@app.route('/', methods=['POST'])
def makeup_image():
    imageBase64 = request.form['image']
    r,g,b,a = request.form['r'], request.form['g'], request.form['b'],request.form['a']
    r = int(r)
    g = int(g)
    b = int(b)
    a = int(a)
    if not imageBase64 or (r < 0 or r > 255) or (g < 0 or g > 255) or (b < 0 or b > 255) or (a < 0 or a > 255):
        result = {
            "errno": 1
        }
        return jsonify(result)
    
    imageData = base64.urlsafe_b64decode(imageBase64)
    loadedImage = loadImageData(imageData)
    finishedImage = makeup.putMakeupOn(loadedImage,r,g,b,a)
    finishedImageReturn = base64.standard_b64encode(imageToBytes(finishedImage)).decode()
    result = {
        "errno": 0,
        "data": finishedImageReturn
    }
    return jsonify(result)


