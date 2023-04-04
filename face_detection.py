from PIL import Image, ImageOps 
import numpy as np
import cv2
import argparse
import json
import sys
from pathlib import Path
from keras_facenet import FaceNet
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


parser = argparse.ArgumentParser()
parser.add_argument('--image_path', default="", type=str)
parser.add_argument('--debug', default=False, type=bool)
args = parser.parse_args()
embedder = FaceNet()


def face_detection(image_path, debug=False):
    """
    find faces on the provided image
    @parameters
    image_path: string
        path to the image
    debug: bool
        if debug mode on
    @return 
    results: list
        list of dictionaries 
        containing information about the location 
        and rotation of each detected face 
    """
    

    file_name_wo_ext = Path(image_path).stem
    angle = 90
    rotated_sum = 0
    results = {"faces":[]}
    raw_image = Image.open(image_path)
    max_size = (640, 480)
    raw_image.thumbnail(max_size)
    im = np.array(raw_image.convert('RGB'))
    detections = embedder.extract(im, threshold=0.8)
    if len(detections) == 0:
        while len(detections) == 0:
            raw_image = raw_image.rotate(angle, expand=True)
            im = np.array(raw_image.convert('RGB'))
            rotated_sum += angle
            detections = embedder.extract(im, threshold=0.8)
            if len(detections) > 0:
                break
            elif rotated_sum >= 360:
                print('Faces not found')
                break
                # sys.exit()
    print("Founded {} face(s) in this photograph.".format(len(detections)))
    for embedding in detections:
        # Get the location of each face in this image
        x, y, w, h = embedding['box']
        # margin is added by expanding the bounding box of the detected face region by a certain percentage
        margin = 10
        x -= margin
        y -= margin
        w += 2*margin
        h += 2*margin
        if x < 0:
            w += x
            x = 0
        if y < 0:
            h += y
            y = 0
        # Draw rectangles around the detected faces
        cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # You can access the actual face itself like this:
        face_image = im[y:y+h, x:x+w]
        pil_image = Image.fromarray(im)
        results_dict = {
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "accuracy": embedding['confidence'],
        "rotate": 0 if rotated_sum==0 else 1,
        "rotated_angle": rotated_sum}
        results["faces"].append(results_dict)
    try:
        if debug:
            pil_image.show()
    except Exception as e:
        print("Faces not found")
    with open(f"{file_name_wo_ext}.json", "w") as temp:
        json.dump(results , temp)
    return results

if __name__ == '__main__':
    if len(args.image_path) > 0:
        face_detection(args.image_path, args.debug)
    else:
        print("Please provide path to the image")
