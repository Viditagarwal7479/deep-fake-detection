import os
import gc
import cv2
import dlib
import csv
from torchvision import transforms
from PIL import Image
from tqdm.notebook import tqdm
import torch
import torchvision
from dataPreprocessing import *

print(dlib.DLIB_USE_CUDA)  # To check whether dlib is using GPU or not


def classify_video(video_path):
    global video_frames
    video = cv2.VideoCapture(video_path)
    frame_num = 0
    while 1:
        ok, frame = video.read()
        if not ok:
            break
        frame_num += 1
        if frame_num & 1:  # Taking alternate frames
            continue
        height, width = frame.shape[:2]
        # Face detector based on HOG and Linear SVM
        face_detector = dlib.get_frontal_face_detector()
        faces = face_detector(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1)
        if len(faces):  # To ensure that there is at least one face in the frame
            # If multiple faces are detected then only use the largest face
            face = faces[0]
            box_coordinate = bounding_box(face, width, height)
            # Cropping out the face from the whole frame
            face = frame[box_coordinate[1]:box_coordinate[3],
                   box_coordinate[0]:box_coordinate[2]]
            # Converting the color space from BGR to RGB
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = Image.fromarray(face)  # Converting the image to PIL format
            # Applying the transformation to the image so that it can be passed into the model
            face = transform(face)
            face = face.unsqueeze(0)  # Adding the batch dimension
            # Concatenating the image to the batch
            video_frames = torch.cat((video_frames, face))
    # Passing the frames in the video as a batch of 16 images to the model (this value can be changed based upon RAM)
    # Removing the first image which was a dummy to create the rest
    video_frames = video_frames[1:]
    batch_size = 16
    n1 = 0
    n2 = batch_size
    # Number of frames upon which prediction will be made
    num_frames_batch = video_frames.size()[0]
    # Creating a tensor to store the predictions
    prediction = torch.Tensor([1])
    che = True
    while che:
        # Passing the frames as batch until required images are there
        if n2 >= num_frames_batch:
            n2 = num_frames_batch
            che = False
        with torch.no_grad():
            output = model(video_frames[n1:n2].cuda())
        # Prediction made upon all the frames
        prediction = torch.cat((prediction, torch.max(output, 1)[1].cpu()))
        del output
        gc.collect()
        n1 = n2
        n2 += batch_size
    real = torch.sum(prediction)  # number of frames detected as real
    fake = prediction.size()[0] - real  # number of frames detected as fake
    ans[video_path.split('/')[-1]] = (fake, real)


ans = {}

video_path = "test/"  # Path to the test videos
l = os.listdir(video_path)
pbar = tqdm(total=len(l))
c = 0
for file in l:
    # Creating a dummy image to start the batch
    video_frames = torch.zeros(size=(1, 3, 299, 299))
    classify_video(video_path + str(file))
    pbar.update(1)
pbar.close()

model = torchvision.models.inception_v3(pretrained=True)
model.dropout = torch.nn.Dropout(p=0.5, inplace=True)
model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=2)

model_path = "InceptionV3.pt"  # Path to the saved model

# As we saved the model state when it was in GPU memory so by default it gets loaded in GPU memory

if torch.cuda.is_available():
    model.load_state_dict(torch.load(model_path))
else:
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Saving the results as a csv file
with open("submission.csv", 'w', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "label"])
    for i in ans:
        writer.writerow([i, 0 if ans[i][0] > ans[i][1] else 1])
