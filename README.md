# FaceNet-with-TripletLoss


My implementation for face recognition using FaceNet model and Triplet Loss. I like to implement different deep learning models architectures. I also like to read about applications and implementations of deep learning models. I have typed this code in my free time as a self learning exercise. So, if you run into some performane issue, i am not an expert , i won't be able to help you. I also don't have hardware to extensively test a heavy model like FaceNet. So, hyperparameters are not tuned at all. Only thing I can assure you is that this implementation works.


> [Medium post](https://medium.com/@mohitsaini_54300/train-facenet-with-triplet-loss-for-real-time-face-recognition-a39e2f4472c3)

### Dependencies
* python 3.6
* tensorflow v1.11
* keras
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


3. Run `train_triplet.py` to train the model. Make changes (if you want) in `parameters.py` to adjust training parameters.

4. Run `webcamFaceRecoMulti.py` to recognize faces in real time. Note- Our dataset must have some images for this script to work.


## Known issues/limitations:
* The dataset must contain the images of atleast two different people to train the model. You can define your own generator function to train it for single person
* Generator function might be slow when dataset has images of fewer number of people.
* Code needs some refactoring.


## Note:
* It is not state of the art technique. So, dont't expect much from it.

* Model is trained using triplet loss. According to experiments it is recommended to chose `positive`, `negative` and `anchor` images carefully/manually for better results. Here I used a generator which selects images for `positive`, `negative` and `anchor` randomly (I'm Lazy af). To know more about this I recommend you to watch [this](https://youtu.be/d2XB5-tuCWU?list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF) video.


## Refrences 
* FaceNet: A Unified Embedding for Face Recognition and Clustering : https://arxiv.org/abs/1503.03832.
* Deepface paper https://www.cs.toronto.edu/~ranzato/publications/taigman_cvpr14.pdf.
* deeplearning.ai 's assignments.
* https://github.com/davidsandberg/facenet
