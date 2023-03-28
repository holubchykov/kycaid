from PIL import Image
import face_recognition
import numpy as np
import cv2
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', default="", type=str)
args = parser.parse_args()


def face_detection(image_path):
    """
    find faces on the provided image
    @parameters
    image_path: string
        path to the image
    @return 
    results: list
        list of dictionaries 
        containing information about the location 
        and rotation of each detected face 
    """
    angle = 90
    rotated_sum = 0
    results = {"faces":[]}
    raw_image = Image.open(image_path)
    im = np.array(raw_image.convert('RGB'))
    face_locations = face_recognition.face_locations(im)
    if len(face_locations) == 0:
        while len(face_locations) == 0:
            raw_image = raw_image.rotate(angle, expand=True)
            im = np.array(raw_image.convert('RGB'))
            rotated_sum += angle
            face_locations = face_recognition.face_locations(im)
            if len(face_locations) > 0:
                break
    print("Founded {} face(s) in this photograph.".format(len(face_locations)))
    for face_location in face_locations:
        # Get the location of each face in this image
        top, right, bottom, left = face_location
        # Draw rectangles around the detected faces
        cv2.rectangle(im, (left, top), (right, bottom), (0, 255, 0), 2)
        # You can access the actual face itself like this:
        face_image = im[top:bottom, left:right]
        pil_image = Image.fromarray(im)
        results_dict = {
        "x": left,
        "y": top,
        "width": right - left,
        "height": bottom - top,
        "rotate": 0 if rotated_sum==0 else 1,
        "rotated_angle": rotated_sum}
        results["faces"].append(results_dict)
    print(results["faces"])
    pil_image.show()
    with open("faces.json", "w") as temp:
        json.dump(results , temp)
    return results

if __name__ == '__main__':
    if len(args.image_path) > 0:
        face_detection(args.image_path)
    else:
        print("Please provide path to the image")
