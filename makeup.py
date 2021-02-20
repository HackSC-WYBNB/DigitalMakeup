from PIL import Image, ImageDraw
from numpy import asarray
import face_recognition
import numpy as np
import PIL
import blend_modes

def singlePointMakeup(pixel, r, g, b, a):
    pixelR, pixelG, pixelB, pixelA = pixel
    newR = pixelR * r / 255
    newG = pixelG * r / 255
    newB = pixelB * r / 255
    newA = pixelA * r / 255
    return (newR,newG,newB,newA)

def putMakeupOn(faceImage, r, g, b, a):
    """
    This function reads from a face image object(already loaded), 
    and puts color RGBA on the lips, returns the altered image
    """
    # Load the jpg 
    # directory might change

    # Find facial features
    faceImageObj = Image.fromarray(faceImage)
    faceImageObj = faceImageObj.convert('RGBA')
    faceImage4L = np.array(faceImageObj)

    face_features = face_recognition.face_landmarks(faceImage)
    
    makeup = Image.new('RGBA',faceImageObj.size,(255,255,255,255))
    for ff in face_features:
        d = ImageDraw.Draw(makeup, 'RGBA')

        d.polygon(ff["top_lip"], fill=(r, g, b, a))
        d.polygon(ff["bottom_lip"], fill=(r, g, b, a))
        d.line(ff["top_lip"], fill=(r, g, b, a), width=3)
        d.line(ff["bottom_lip"], fill=(r, g, b, a), width=3)

    makeupNP = np.array(makeup)
    makeupNP_Float = makeupNP.astype(float)
    faceNP_Float = faceImage4L.astype(float)
    concatedFaceImageNP_Float = blend_modes.multiply(faceNP_Float,makeupNP_Float,0.7)
    concatedFaceImageNP = np.uint8(concatedFaceImageNP_Float)
    
    return np.array(Image.fromarray(concatedFaceImageNP).convert('RGB'))

"""
def main():
    image = np.array(PIL.Image.open('1.jpeg',formats=['JPEG','PNG','BITMAP']))
    image = putMakeupOn(image, 150, 0, 0, 64)
    image.save("output.jpeg")
"""