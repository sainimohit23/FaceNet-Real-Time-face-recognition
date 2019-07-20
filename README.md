# FaceNet-with-TripletLoss


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


2. Use `align_dataset_mtcnn.py` to prepare our dataset for training. Run the following command:

```python align_dataset_mtcnn.py ./YOUR_DIRECTIORY_CONTAINING_DATA ./cropped```

example:

![Screenshot_99](https://user-images.githubusercontent.com/26195811/61582654-07025100-ab4b-11e9-9775-60125bfee825.png)


3. Run `train_triplet.py` to train the model. Make changes in `parameters.py` to adjust training parameters.

4. Run `webcamFaceRecoMulti.py` to recognize faces in real time. Note- Our dataset must have some images for this script to work.

![ezgif com-resize](https://user-images.githubusercontent.com/26195811/50422107-10556480-086d-11e9-9016-e8886aca4140.gif)


## Known issues:
* The dataset must contain the images of atleast two different people.
* Generator function might be slow when dataset has images of fewer number of people.


## Note:
* It is not state of the art technique. So, dont't expect much from it.

* Model is trained using triplet loss. According to experiments it is recommended to chose `positive`, `negative` and `anchor` images carefully/manually for better results. Here I used a generator which selects images for `positive`, `negative` and `anchor` randomly (I'm Lazy af). To know more about this I recommend you to watch [this](https://youtu.be/d2XB5-tuCWU?list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF) video.


## Refrences 
* FaceNet: A Unified Embedding for Face Recognition and Clustering : https://arxiv.org/abs/1503.03832.
* Deepface paper https://www.cs.toronto.edu/~ranzato/publications/taigman_cvpr14.pdf.
* deeplearning.ai 's assignments.
* https://github.com/davidsandberg/facenet
