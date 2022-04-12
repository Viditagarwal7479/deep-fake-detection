from pydoc import cli
import cv2
import os
import dlib
from tqdm import tqdm
from dask.distributed import Client

dlib.DLIB_USE_CUDA  # To make dlib use GPU if available


def bounding_box(face, width, height):
    scale = 1.3  # Scaling up
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    # To get the side of the square box around the face with scaling
    side = int(max(x2 - x1, y2 - y1) * scale)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    # To ensure that the box is inside the frame
    x1 = max(int(cx - side // 2), 0)
    y1 = max(int(cy - side // 2), 0)
    x2 = min(int(cx + side // 2), width)
    y2 = min(int(cy + side // 2), height)
    return x1, y1, x2, y2


def video_to_image(video_path):
    global count
    video = cv2.VideoCapture(video_path)
    frame_num = 0
    while 1:
        ok, frame = video.read()
        if not ok:
            break
        frame_num += 1

        if frame_num % 20 != 0:  # Only taking 1 frame in 20 frame
            continue
        height, width = frame.shape[:2]
        # Face detector based on HOG and Linear SVM
        face_detector = dlib.get_frontal_face_detector()
        faces = face_detector(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1)
        # To ensure that there is at least one face in the frame
        if len(faces):
            # If multiple faces are detected then only use the largest face
            face = faces[0]
            box_coordinate = bounding_box(face, width, height)
            face = frame[
                   box_coordinate[1]:box_coordinate[3],
                   box_coordinate[0]:box_coordinate[2]
                   ]  # Cropping out the face from the whole frame
            return face
        else:
            frame_num -= 1


if __name__ == "__main__":
    client = Client()
    print(f"Dask Dashboard Link: {client.dashboard_link}")

    os.makedirs("images", exist_ok=True)
    os.makedirs("images/train", exist_ok=True)
    os.makedirs("images/test", exist_ok=True)
    os.makedirs("images/train/fake", exist_ok=True)
    os.makedirs("images/test/fake", exist_ok=True)
    os.makedirs("images/train/real", exist_ok=True)
    os.makedirs("images/test/real", exist_ok=True)

    fake_video_folder = "videos/fake/"
    l = os.listdir(fake_video_folder)
    a = client.map(video_to_image, l)
    a = client.gather(a)
    s = 0
    count = 0
    for i in tqdm(a):
        try:
            if count % 5:
                cv2.imwrite("images/train/fake/{}.jpg".format(count), i)
            else:
                cv2.imwrite("images/test/fake/{}.jpg".format(count), i)
            s += 1
        except Exception as e:
            pass
        count += 1
    print(f"FAKE:\n{s} videos saved successfully out of {count} videos")

    real_video_folder = "videos/real/"
    l = os.listdir(real_video_folder)
    a = client.map(video_to_image, l)
    a = client.gather(a)
    s = 0
    count = 0
    for i in tqdm(a):
        try:
            if count % 5:
                cv2.imwrite("images/train/real/{}.jpg".format(count), i)
            else:
                cv2.imwrite("images/test/real/{}.jpg".format(count), i)
            s += 1
        except Exception as e:
            pass
        count += 1
    print(f"REAL:\n{s} videos saved successfully out of {count} videos")

    client.close()
    client.shutdown()
