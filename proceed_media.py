from PIL import Image
import numpy as np
import time
import cv2
import argparse
import json
import sys
import emotions_model
import gender
import age
from pathlib import Path
from keras_facenet import FaceNet
import tensorflow as tf
from keras.preprocessing import image
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default="image_processing", type=str)
parser.add_argument('--video_path', default="", type=str)
parser.add_argument('--img_path', default="", type=str)
parser.add_argument('--debug', default=False, type=bool)
args = parser.parse_args()
embedder = FaceNet()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
gender_model = gender.loadModel()
emo_model = emotions_model.get_emo_model()
age_model = age.loadModel()
font = cv2.FONT_HERSHEY_SIMPLEX
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}


def transform_face_array2gender_face(face_array, grayscale=False, target_size=(224, 224)):
    detected_face = face_array
    if grayscale:
        detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)
    detected_face = cv2.resize(detected_face, target_size)
    img_pixels = image.img_to_array(detected_face)
    img_pixels = np.expand_dims(img_pixels, axis=0)
    img_pixels /= 255
    return img_pixels


def process_video(video_path, output_video_path='output_video.MOV'):
    """
    Find faces on the provided video. Detect emotion, age and gender.

    Args:
        video_path (string): Video to predict features.
        output_video_path (string): Path for the proceeded video save to .

    Returns:
        Proceeded video: A video with bounding boxes and feature predictions for all faces
    """
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    writer = cv2.VideoWriter(output_video_path, fourcc, int(fps), (int(width), int(height)))
    idx = 0
    while True:
        ret, frame = cap.read()
        if idx >= total_frames:
            break
        if frame is None:
            print('pass img_raw is empty, status:')
            time.sleep(1)
            continue
        idx += 1
        if idx % 3 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detections = embedder.extract(frame, threshold=0.8)
            for embedding in detections:
                x, y, w, h = embedding['box']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                face_img = frame[y:y + h, x:x + w]
                roi_gray = gray[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                prediction = emo_model.predict(cropped_img)
                maxindex = int(np.argmax(prediction))
                cv2.putText(frame, f"Emotion: {emotion_dict[maxindex]}", (x, y + h + 20), font, 0.5,
                            (255, 255, 255), 2, cv2.LINE_AA)
                gender_img = transform_face_array2gender_face(face_img)
                gender_res = gender_model.predict(gender_img)[0, :]
                if np.argmax(gender_res) == 0:
                    result_gender = "Woman"
                elif np.argmax(gender_res) == 1:
                    result_gender = "Man"
                cv2.putText(frame, f"Gender: {result_gender}", (x, y + h + 35), font, 0.5,
                            (255, 255, 255), 2, cv2.LINE_AA)
                age_res = age_model.predict(gender_img)[0, :]
                result_age = age.findApparentAge(age_res)
                label = f"Age: {int(result_age)}"
                cv2.putText(frame, label, (x, y + h + 50), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                if result_gender == 'Woman':
                    cv2.putText(frame, 'Melina Grace, actress', (x, y - 10), font, 0.5, (255, 255, 255),
                                2, cv2.LINE_AA)
            writer.write(frame)
        cv2.waitKey(1)
        print("%d of %d" % (idx, total_frames))
    writer.release()


def process_img(img_path):
    """
    Find faces on the provided image. Detect emotion, age and gender.

    Args:
        img_path (string): Image to predict features.

    Returns:
        result (list): A list that contains dictionary with faces bounding boxes and features predictions
    """
    results = []
    raw_image = Image.open(img_path)
    max_size = (640, 480)
    raw_image.thumbnail(max_size)
    im = np.array(raw_image.convert('RGB'))
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    detections = embedder.extract(im, threshold=0.8)
    if len(detections) == 0:
        print("Faces weren't found on the provided image")
        sys.exit()
    for embedding in detections:
        x, y, w, h = embedding['box']
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face_img = im[y:y + h, x:x + w]
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = emo_model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        cv2.putText(im, f"Emotion: {emotion_dict[maxindex]}", (x, y + h + 20), font, 0.5,
                    (255, 255, 255), 2, cv2.LINE_AA)
        gender_img = transform_face_array2gender_face(face_img)
        gender_res = gender_model.predict(gender_img)[0, :]
        if np.argmax(gender_res) == 0:
            result_gender = "Woman"
        elif np.argmax(gender_res) == 1:
            result_gender = "Man"
        cv2.putText(im, f"Gender: {result_gender}", (x, y + h + 35), font, 0.5,
                    (255, 255, 255), 2, cv2.LINE_AA)
        age_res = age_model.predict(gender_img)[0, :]
        result_age = age.findApparentAge(age_res)
        label = f"Age: {int(result_age)}"
        cv2.putText(im, label, (x, y + h + 50), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        res_dict = {"bounding_box": embedding['box'],
                    "Emotion": emotion_dict[maxindex],
                    "Age": int(result_age),
                    "Gender": result_gender}
        results.append(res_dict)
    pil_image = Image.fromarray(im)
    if args.debug:
        pil_image.show()
    file_name_wo_ext = Path(img_path).stem
    with open(f"{file_name_wo_ext}.json", "w") as temp:
        json.dump(results, temp)
    return results


if __name__ == '__main__':
    if args.mode.startswith('image'):
        if len(args.img_path) > 0:
            process_img(args.img_path)
        else:
            print("Please provide path to the image")
    elif args.mode.startswith('video'):
        if len(args.video_path) > 0:
            process_video(args.video_path)
        else:
            print("Please provide path to the video")
    else:
        print("Please choose the right mode")


