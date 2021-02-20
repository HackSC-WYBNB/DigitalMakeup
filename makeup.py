from PIL import Image, ImageDraw
from numpy import asarray
import face_recognition
import numpy as np
import PIL
import blend_modes
import math

def _compose_alpha(img_in, img_layer, opacity):
    """Calculate alpha composition ratio between two images.
    """

    comp_alpha = np.minimum(img_in[:, :, 3], img_layer[:, :, 3]) * opacity
    new_alpha = img_in[:, :, 3] + (1.0 - img_in[:, :, 3]) * comp_alpha
    np.seterr(divide='ignore', invalid='ignore')
    ratio = comp_alpha / new_alpha
    ratio[ratio == np.NAN] = 0.0
    return ratio

def linear_burn(img_in, img_layer, opacity, disable_type_checks: bool = False):

    if not disable_type_checks:
        _fcn_name = 'multiply'
        blend_modes.assert_image_format(img_in, _fcn_name, 'img_in')
        blend_modes.assert_image_format(img_layer, _fcn_name, 'img_layer')
        blend_modes.assert_opacity(opacity, _fcn_name)

    img_in_norm = img_in / 255.0
    img_layer_norm = img_layer / 255.0

    ratio = _compose_alpha(img_in_norm, img_layer_norm, opacity)

    comp = np.clip(img_layer_norm[:, :, :3] + img_in_norm[:, :, :3] - 1.0, 0.0, 1.0)

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp * ratio_rs + img_in_norm[:, :, :3] * (1.0 - ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_in_norm[:, :, 3])))  # add alpha channel and replace nans
    return img_out * 255.0

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
        
        widthToDraw = 8
        for i in range(widthToDraw):
            lineAlpha = round(a * (widthToDraw - i))
            d.line(ff["top_lip"], fill=(r, g, b, lineAlpha), width=i)
            d.line(ff["bottom_lip"], fill=(r, g, b, lineAlpha), width=i)

    makeupNP = np.array(makeup)
    makeupNP_Float = makeupNP.astype(float)
    faceNP_Float = faceImage4L.astype(float)

    filter = blend_modes.multiply

    concatedFaceImageNP_Float = filter(faceNP_Float,makeupNP_Float,0.8)
    concatedFaceImageNP = np.uint8(concatedFaceImageNP_Float)
    
    return np.array(Image.fromarray(concatedFaceImageNP).convert('RGB'))

"""
def main():
    image = np.array(PIL.Image.open('1.jpeg',formats=['JPEG','PNG','BITMAP']))
    image = putMakeupOn(image, 150, 0, 0, 64)
    image.save("output.jpeg")
"""