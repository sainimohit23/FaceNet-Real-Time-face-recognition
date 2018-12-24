import os
import cv2
import numpy as np


paths = {'mohit' : "./cropped/mohit/",
        'mummy' : "./cropped/mummy/"}
faces = ['mohit', 'mummy']


negPath = './cropped/myimages/'
negImages = []

images = []
for key in paths.keys():
    li = []
    for img in os.listdir(paths[key]):
        img1 = cv2.imread(paths[key]+img)
        img2 = img1[...,::-1]
        li.append(np.around(np.transpose(img2, (2,0,1))/255.0, decimals=12))
    images.append(li)

for img in os.listdir(negPath):
    img1 = cv2.imread(negPath+img)
    img2 = img1[...,::-1]
    negImages.append(np.around(np.transpose(img2, (2,0,1))/255.0, decimals=12))
    

for i in range(len(images)):
    images[i] = np.array(images[i])
negImages = np.array(negImages)


def batch_generator(batch_size = 64):
    while True:
        a= np.random.randint(0, len(images), 1)
        pos = images[a[0]][np.random.choice(len(images[a[0]]), batch_size)]
        neg = negImages[np.random.choice(len(negImages), batch_size)]
        anc = images[a[0]][np.random.choice(len(images[a[0]]), batch_size)]

        x_data = {'anchor': anc,
                  'anchorPositive': pos,
                  'anchorNegative': neg
                  }
        
        yield (x_data, np.zeros((batch_size, 2, 1)))
        
def batch_generator2(batch_size = 64):
    while True:
        a= np.random.randint(0, len(images), 1)
        b= np.random.randint(0, len(images), 1)
        while b==a:
            b= np.random.randint(0, len(images), 1)
            
        pos = images[a[0]][np.random.choice(len(images[a[0]]), batch_size)]
        neg = images[b[0]][np.random.choice(len(images[b[0]]), batch_size)]
        anc = images[a[0]][np.random.choice(len(images[a[0]]), batch_size)]

        x_data = {'anchor': anc,
                  'anchorPositive': pos,
                  'anchorNegative': neg
                  }
        
        yield (x_data, np.zeros((batch_size, 2, 1)))
    