from keras_facenet import FaceNet
import argparse
import json
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
parser = argparse.ArgumentParser()
parser.add_argument('--image1_path', default="", type=str)
parser.add_argument('--image2_path', default="", type=str)
parser.add_argument('--debug', default=False, type=bool)
args = parser.parse_args()
embedder = FaceNet()


def face_matching(img1, img2):
    """
    Calculate distance between each embedding from the first image with each embedding from the second image.

    Args:
        img1 (string): The first photo for comparison.
        img2 (string): The second photo for comparison.

    Returns:
        matching_results (list): A list that contains dictionary with calculated
        distance and bounding box from the each image
    """
    matching_results = []
    detections_one, crops_one = embedder.crop(img1)
    detections_two, crops_two = embedder.crop(img2)
    embedding1 = embedder.embeddings(crops_one)
    embedding2 = embedder.embeddings(crops_two)
    for i, arr in enumerate(embedding1):
        for j, arr2 in enumerate(embedding2):
            distance = embedder.compute_distance(arr, arr2)
            if distance < 0.5:
                res = {'distance': distance,
                       'bounding_box_first_photo': detections_one[i]['box'],
                       'bounding_box_second_photo': detections_two[j]['box']
                       }
                matching_results.append(res)
    return matching_results


if __name__ == '__main__':
    if len(args.image1_path) > 0 and len(args.image2_path) > 0:
        res = face_matching(args.image1_path, args.image2_path)
        with open('results.json', 'w') as json_file:
            json.dump(res, json_file)
        print(f'Results:\n {res}')
    else:
        print("Please provide path to the image")
