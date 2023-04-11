# kycaid face detection
It has been tested on Linux Ubuntu
<br />For install all necessary requirements:
<br />
<br />**pip install -r requirements.txt**
<br />
### to run:
python face_detection.py --image_path="/path/to/image.jpeg"
<br />
python face_matching.py --image1_path="/path/to/image1.jpeg" --image2_path="/path/to/image2.jpeg"


### Result:
#### face_detection
As a result, it returns a json file with faces coordinates, rotated image or not (1 or 0), if it's rotated also return rotated angle.
<br /> Also it shows image with bounding boxes
<br />
#### face_matching
As a result, it returns a json file with founded matching faces on the images. Results contains calculated distance between embeddings and bounding box from the each image.
