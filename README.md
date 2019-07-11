# FaceNet-with-TripletLoss

NOTE: Don't use the code. The implementation is not correct. I will fix it in few days.


My implementation for face recognition using FaceNet model and Triplet Loss.

> [Medium post](https://medium.com/@mohitsaini_54300/train-facenet-with-triplet-loss-for-real-time-face-recognition-a39e2f4472c3)

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

<!--- ![ezgif com-resize](https://user-images.githubusercontent.com/26195811/50422107-10556480-086d-11e9-9016-e8886aca4140.gif) --->


## Note:
* It is not state of the art technique. So, dont't expect much from it.

* Model is trained using triplet loss. According to experiments it is recommended to chose `positive`, `negative` and `anchor` images carefully/manually for better results. Here I used a generator which selects images for `positive`, `negative` and `anchor` randomly (I'm Lazy af). To know more about this I recommend you to watch [this](https://youtu.be/d2XB5-tuCWU?list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF) video.

* While training the model it will show very low loss straight away from the beginning. Don't fall for that. I will later include some reliable metric to get idea of model's accuracy on data while training.


## Refrences 
* FaceNet: A Unified Embedding for Face Recognition and Clustering : https://arxiv.org/abs/1503.03832.
* Deepface paper https://www.cs.toronto.edu/~ranzato/publications/taigman_cvpr14.pdf.
* deeplearning.ai 's assignments.



