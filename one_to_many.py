from keras_facenet import FaceNet
import argparse
import json
import tensorflow as tf
import os
from PIL import Image, ImageOps 
import numpy as np
import cv2
from pathlib import Path
import glob
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
parser = argparse.ArgumentParser()
parser.add_argument('--image_path', default="", type=str)
parser.add_argument('--db_path', default="/media/pavlo_holubchykov/1TB/kycaid_one_to_many/db_faces", type=str)
args = parser.parse_args()
embedder = FaceNet()

detections_main, crops_main = embedder.crop('/home/pavlo_holubchykov/Pictures/kycaid_test/pasha2.jpg')
embeddings_main = embedder.embeddings(crops_main)
faces_path = "faces.json"


def create_json_w_persons_embeddings(img_dir):

    # Define the file extensions of the images you want to iterate over
    file_ext = "*"  # for example, to iterate over all jpg files

    # Get a list of file paths in the directory that match the file extension
    img_paths = glob.glob(os.path.join(img_dir, file_ext))

    dictionary_of_face_embeddigs = {}
    # Iterate over the image paths and do something with each image
    for path in img_paths:
        file_name_wo_ext = Path(path).stem
        detections_one, crops_one = embedder.crop(path)
        embeddings = embedder.embeddings(crops_one)
        if len(embeddings) > 1:
            max_square = 0
            for i, embedding in enumerate(detections_one):
                bb_square = embedding['box'][2] * embedding['box'][3]
                if bb_square > max_square:
                    max_square = bb_square
                    main_person = i
            dictionary_of_face_embeddigs[file_name_wo_ext] = list(embeddings[main_person])
        else:
            dictionary_of_face_embeddigs[file_name_wo_ext] = list(embeddings[0])
    with open('faces.json', 'w') as json_file:
        json.dump(dictionary_of_face_embeddigs, json_file, default=float)
    return dictionary_of_face_embeddigs


def find_person_in_db(embeddings_main_person, dictionary_of_face_embeddigs):
    finded_person = ''
    for person in dictionary_of_face_embeddigs:
        distance = embedder.compute_distance(embeddings_main_person, dictionary_of_face_embeddigs[person])
        if distance < 0.5:
            finded_person = person
    if finded_person:
        print(f"Person on the photo is {finded_person}")
    else:
        print("The person on the image is not in the database")


if __name__ == '__main__':
    if len(args.image_path) > 0 and len(args.db_path) > 0:
        # Check if the file exists
        if not os.path.isfile(faces_path):
            dictionary_of_face_embeddigs = create_json_w_persons_embeddings(args.db_path)
            find_person_in_db(embeddings_main, dictionary_of_face_embeddigs)
        else:
            with open(faces_path, 'r') as f:
                dictionary_of_face_embeddigs = json.load(f)
            find_person_in_db(embeddings_main, dictionary_of_face_embeddigs)
    else:
        print("Please provide path to the image")
