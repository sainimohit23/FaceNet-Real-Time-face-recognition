# FaceNet-with-TripletLoss

My implementation for face recognition using FaceNet model and Triplet Loss.

### Dependencies
* keras
* numpy
* time
* os
* openCV

### Usage
1. Create a dataset of faces for each person and arrange them in below order.
```
root folder  
│
└───Person 1
│   │───IMG1
│   │───IMG2
│   │   ....
└───Person 2
|   │───IMG1
|   │───IMG2
|   |   ....
```

2. Run `align_dataset_mtcnn.py` to align faces. This code is taken from [facenet](https://github.com/davidsandberg/facenet). Usage example:

![screenshot_43](https://user-images.githubusercontent.com/26195811/50400027-990acc80-07a9-11e9-860c-a20ab53bc5a8.png)

3. Edit `path` dictionary and `face` list in `generator_utils.py` according to your data.

4. Run `train_triplet.py` to train the model. Adjust parameters accordingly.

5. Run `webcamFaceRecoMulti.py` to recognize faces in real time.

![ezgif com-resize](https://user-images.githubusercontent.com/26195811/50422107-10556480-086d-11e9-9016-e8886aca4140.gif)

## Refrences 
* FaceNet: A Unified Embedding for Face Recognition and Clustering : https://arxiv.org/abs/1503.03832.
* Deepface paper https://www.cs.toronto.edu/~ranzato/publications/taigman_cvpr14.pdf.
* deeplearning.ai 's assignments.



