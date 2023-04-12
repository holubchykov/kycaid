# kycaid features
It has been tested on Linux Ubuntu
<br />For install all necessary requirements:
<br />
<br />**pip install -r requirements.txt**
<br />
### to run:
python face_detection.py --mode="image" --img_path="/path/to/image.jpeg"
<br />
python face_detection.py --mode="video" --video_path="/path/to/video.mov


### Result:
#### Photo
As a result, it returns a json file with faces bounding boxes and features, such as: emotion, age, gender.
It also shows proceeded image if add flag --debug=True
<br />
#### video
As a result, it returns a proceeded video with faces bounding boxes and all mentioned features.
Video saving under the current working directory with name "output_video.mov"
