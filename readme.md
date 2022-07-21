# DeepFake Detection

## Approach:

- The face was extracted and passed into the model as the background is static i.e. it will be the same for a video that is real or a deepfake. So passing unnecessary information into the model isn't feasible.

- Here dlib.get_frontal_face_detector() was used which is based upon HOG (Histograms\
 of Oriented Gradients) and Linear SVM. Other functions such as dlib cnn_face_detection_model_v1("model_path") where we can pass pretrained cnn model like MMOD (Max Margin Object Detection) but keeping time and resources under view HOG + Linear SVM based face detection was most suitable.
 
- Only taking 1 frame in 20 frame to reduce the processing time, dataset size, ensuring that model doesn't start to classify based on 
  the face as in a video the nearby frames look very much the same so make a good dataset and we have about double the fake video dataset than real 
  video. A good model needs a good dataset.
  
- Then a InceptionV3 model was used a binary classifier to categorise real and fake images. So the video have more number of fake frames will be a deep fake video.


## How to Use:

- In dataPreprocessing.py change the paths of the folder where the real and fake videos have been downloaded.
- In training.py and submission.py ensure that the path is coorect, and change the batch size based on hardware available.

## Results

- Mean F1-Score of 0.9871.

## Kaggle Jupyter Notebook Link
- https://www.kaggle.com/viditagarwal112/deepfake-detection-inceptionv3

## About the competeion and dataset
- https://www.kaggle.com/c/deepfake-detection/
