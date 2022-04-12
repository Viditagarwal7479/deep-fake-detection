# DeepFake Detection

## Approach:

- The face was extracted and passed into the model as the background is static i.e. it will be the same for a video that is real or a deepfake. So passing unnecessary information into the model isn't feasible.

- Here dlib.get_frontal_face_detector() was used which is based upon HOG (Histograms\
 of Oriented Gradients) and Linear SVM. Other functions such as dlib cnn_face_detection_model_v1("model_path") where we can pass pretrained cnn model like MMOD (Max Margin Object Detection) but keeping time and resources under view HOG + Linear SVM based face detection was most suitable.
- Only taking 1 frame in 20 frame to reduce the processing time, dataset size, ensuring that model doesn't start to classify based on 
  the face as in a video the nearby frames look very much the same so make a good dataset and we have about double the fake video dataset than real 
  video. A good model needs a good dataset.
- I have used Adam optimizer but SGD with momentum is also a nice candidate as an optimizer but in an article, I read that Adam is faster but compromises a bit upon convergence as compared to SGD with momentum.
Final results might vary a bit as the dataset was shuffled between epochs (shuffle=True) and based upon some random parameters at the time of initialization.


## How to Use:

- In dataPreprocessing.py change the paths of the folder where the real and fake videos have been downloaded.
- In training.py and submission.py ensure that the path is coorect, and change the batch size based on hardware available.

## Hardware Used:

- intel xeon gold 6248r (4 cores)
- 16GB RAM
- Nvidia Tesla T4 (16 GB RAM)

## Metrics

- Mean F1-Score of 0.9871.

## About the competeion and dataset
- https://www.kaggle.com/c/deepfake-detection/