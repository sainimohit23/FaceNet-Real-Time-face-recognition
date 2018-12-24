import cv2

# HaarCascade Face Detector
class faceDetector: 
    
    def __init__(self, path):
        self.faceCascade = cv2.CascadeClassifier(path)
        
    def detect(self, image, scaleFactor= 1.02, minNeb = 15, minSize = (100, 100)):
            
        rects =  self.faceCascade.detectMultiScale(image, scaleFactor= scaleFactor,
                                                   minNeighbors= minNeb, minSize= minSize,
                                                   flags= cv2.CASCADE_SCALE_IMAGE)
        
        return rects





