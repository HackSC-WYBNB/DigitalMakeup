from PIL import Image, ImageDraw
from numpy import asarray
import face_recognition

def putMakeupOn(faceImage, r, g, b, a):
    """
    This function reads from a face image object(already loaded), 
    and puts color RGBA on the lips, returns the altered image
    """

    # Load the jpg 
    # directory might change
    image = asaray(faceImage)

    # Find facial features
    face_features = face_recognition.face_landmarks(image)

    makeup = Image.fromarray(image)
    for ff in face_features:
        d = ImageDraw.Draw(makeup, 'RGBA')

        d.polygon(ff["top_lip"], fill=(r, g, b, a))
        d.polygon(ff["bottom_lip"], fill=(r, g, b, a))
        d.line(ff["top_lip"], fill=(r, g, b, a), width=3)
        d.line(ff["bottom_lip"], fill=(r, g, b, a), width=3)

    faceImage = makeup
    return faceImage

# def main():
#     image = Image.open("/Users/lolxu/Desktop/HackSC/me.jpg")
#     image = putMakeupOn(image, 150, 0, 0, 64)
#     image.save("/Users/lolxu/Desktop/HackSC/melips.jpg")
