import streamlit as st
import imageio.v2 as imageio
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import os
from imutils import paths
import cv2
import imutils
import dlib
from imutils import face_utils
import shutil

st.title('Lip README')


col1, col2 = st.columns(2)


def create_dir(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print(f"ERROR: creating directory with name {path}")


def save_frame(video_path, save_dir, gap=1):
    #     name = video_path.split("/")[-1].split(".")[0]
    name = 'frames'
    save_path = os.path.join(save_dir, name)
    create_dir(save_path)
    print(save_path)
    print(video_path)

    cap = cv2.VideoCapture(video_path)
    idx = 0

    while True:
        ret, frame = cap.read()

        if ret == False:
            cap.release()
            break

        if idx == 0:
            cv2.imwrite(f"{save_path}/{idx}.png", frame)
        else:
            if idx % gap == 0:
                cv2.imwrite(f"{save_path}/{idx}.png", frame)

        idx += 1


# deleting folders if there are already images stored of previous prediction
if os.path.exists(r'custom\frames'):
    shutil.rmtree(r'custom\frames')
if os.path.exists(r'custom\cropped'):
    shutil.rmtree(r'custom\cropped')

# making empty folders to store images, video and gif
if not os.path.exists(r'custom'):
    os.mkdir(r'custom')
if not os.path.exists(r'custom\frames'):
    os.mkdir(r'custom\frames')
if not os.path.exists(r'custom\cropped'):
    os.mkdir(r'custom\cropped')


def save_uploaded_file(uploadedfile):
    with open(os.path.join(os.path.join('custom'), uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success("Saved file : {}  ...".format(uploadedfile.name))


uploaded_video = st.file_uploader("Upload video", type=['mp4'])

if uploaded_video is not None:
    file_details = {"FileName": uploaded_video.name,
                    "FileType": uploaded_video.type, "FileSize": uploaded_video.size}
    save_uploaded_file(uploaded_video)
    selected_video = uploaded_video


print(uploaded_video.name)

with st.spinner('Wait for it, processing...'):

    save_frames = r"custom"
    video_path = fr"custom\{uploaded_video.name}"

    with col1:
        st.info('This is the uploaded video')
        file_path = video_path

        # Rendering inside of the app
        video = open(file_path, 'rb')
        video_bytes = video.read()
        st.video(video_bytes)

    save_frame(video_path, save_frames, gap=5)

    file_list = os.listdir(r'custom\frames')
    print(file_list)

    def crop_and_save_image(img, img_path, write_img_path, img_name):
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(
            'shape_predictor_68_face_landmarks.dat')
        # load the input image, resize it, and convert it to grayscale

        image = cv2.imread(img_path)
        image = imutils.resize(image, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale image
        rects = detector(gray, 1)
        if len(rects) > 1:
            print("Error")
            return
        if len(rects) < 1:
            print("ERROR: no faces detected")
            return
        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            name, i, j = 'mouth', 48, 68
            # clone = gray.copy()
            (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
            roi = gray[y:y+h, x:x+w]
            roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
            print(write_img_path)
            cv2.imwrite(write_img_path, roi)

    directory = r'custom/frames/'
    dir_temp = r'custom/cropped/'
    for img_name in file_list:
        image = imageio.imread(directory + '' + img_name)
        crop_and_save_image(image, directory + '' + img_name,
                            dir_temp + '' + img_name, img_name)

    from tensorflow.keras.models import Model
    from tensorflow.keras.models import load_model

    savedModel = load_model('lip_model_cnn_lstm.h5')

    max_seq_length = 22
    MAX_WIDTH = 100
    MAX_HEIGHT = 100

    from skimage.transform import resize
    import time
    sequence = []
    for img_name in file_list:
        image = imageio.imread(dir_temp + '/' + img_name)
        image = resize(image, (MAX_WIDTH, MAX_HEIGHT))
        image = 255 * image
        # Convert to integer data type pixels.
        image = image.astype(np.uint8)
        sequence.append(image)
    pad_array = [np.zeros((MAX_WIDTH, MAX_HEIGHT))]
    sequence.extend(pad_array * (max_seq_length - len(sequence)))
    sequence = np.array(sequence)

    def normalize_it(X):
        v_min = X.min(axis=(2, 3), keepdims=True)
        v_max = X.max(axis=(2, 3), keepdims=True)
        X = (X - v_min)/(v_max - v_min)
        X = np.nan_to_num(X)
        return X

    X_test = []
    X_test.append(sequence)
    X_test = np.array(X_test)
    X_test = normalize_it(X_test)
    X_test = np.expand_dims(X_test, axis=4)
    ypred = savedModel.predict(X_test)
    words = ['Begin', 'Choose', 'Connection', 'Navigation',
             'Next', 'Previous', 'Start', 'Stop', 'Hello', 'Web']
    predicted_words = [words[i] for i in np.argmax(ypred, axis=1)]
    print(predicted_words)

    with col2:

        image_dir = r'custom/cropped/'
        image_files = [os.path.join(image_dir, f) for f in os.listdir(
            image_dir) if f.endswith(".png")]
        image_files = sorted(image_files)
        images = []

        for filename in image_files:
            images.append(imageio.imread(filename))

        imageio.mimsave("custom/animation.gif", images, fps=10)

        st.info('This is what the model sees for making prediction ')
        st.image("custom/animation.gif", width=400)


st.info('Predicted word is :  ')

st.subheader(predicted_words[0])
