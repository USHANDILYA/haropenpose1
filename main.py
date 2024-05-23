# import cv2
# import numpy as np
# from extract import get_keypoints as get_keypoints_openpose
# from extract import get_keypoints as get_keypoints_tfpose
# from prepare import preprocess_keypoints
# from tf_pose import common
# from tf_pose.estimator import TfPoseEstimator
# from tf_pose.networks import get_graph_path
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Flatten
# from sklearn.model_selection import train_test_split
# import os

# # Specify the directory containing your dataset
# dataset_dir = "C:\Users\UTKARSH\Desktop\data science\dl\har2\training"

# # List all video files in the dataset directory
# video_files = [os.path.join(dataset_dir, file) for file in os.listdir(dataset_dir) if file.endswith(".mp4")]
# # Step 1: Load the video files
# #video_files = [...]  # List of video file paths

# # Step 2: Extract keypoints
# keypoints = []
# for video_file in video_files:
#     # Extract keypoints using OpenPose or tf-pose
#     # Example:
#     keypoints_openpose = get_keypoints_openpose(video_file)
#     eypoints_tfpose = get_keypoints_tfpose(video_file)
#     keypoints.extend(keypoints_openpose)  # or keypoints_tfpose

# # Step 3: Preprocess keypoints
# preprocessed_keypoints = [preprocess_keypoints(k) for k in keypoints]

# # Step 4: Create dataset
# labels = [...]  # List of corresponding labels for each video/frame

# # Split dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(preprocessed_keypoints, labels, test_size=0.2, random_state=42)

# # Step 5: Define LSTM model
# input_shape = (None, 25, 3)  # Sequence length is variable
# num_classes = 10  # Define your number of classes

# model = Sequential()
# model.add(TimeDistributed(Flatten(), input_shape=input_shape))
# model.add(LSTM(128, return_sequences=False))
# model.add(Dense(num_classes, activation='softmax'))

# model.compile(loss='sparse_categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])

# # Step 6: Train the model
# num_epochs = 50
# model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test))
import os
import cv2
import numpy as np
from extract import get_keypoints as get_keypoints_openpose
from extract import get_keypoints as get_keypoints_tfpose
from prepare import preprocess_keypoints
from tf_pose import common
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Flatten
from sklearn.model_selection import train_test_split

# Specify the root directory containing your dataset
root_dir = r"C:\Users\UTKARSH\Desktop\data science\dl\har2\training"

# Get a list of class folders
class_folders = sorted([folder for folder in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, folder))])

# Initialize lists to store keypoints and corresponding class labels
keypoints = []
class_labels = []

# Iterate through each class folder
for class_folder in class_folders:
    class_folder_path = os.path.join(root_dir, class_folder)
    video_files = [os.path.join(class_folder_path, file) for file in os.listdir(class_folder_path) if file.endswith(".mp4")]
    
    # Extract keypoints from videos in the current class folder
    for video_file in video_files:
        # Extract keypoints using OpenPose or tf-pose
        # Example:
        #keypoints_openpose = get_keypoints_openpose(video_file)
        keypoints_tfpose = get_keypoints_tfpose(video_file)
        keypoints.extend(keypoints_tfpose)  # or keypoints_tfpose
        #keypoints.append(...)  # Append keypoints extracted from current video
        
        # Append class label
        class_labels.extend([class_folder] * len(keypoints_tfpose))

# Step 3: Preprocess keypoints
preprocessed_keypoints = [preprocess_keypoints(k) for k in keypoints]

# Step 4: Create dataset
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(preprocessed_keypoints, class_labels, test_size=0.2, random_state=42)

# Step 5: Define LSTM model
input_shape = (None, 25, 3)  # Sequence length is variable
num_classes = len(set(class_labels))  # Number of unique classes

model = Sequential()
model.add(TimeDistributed(Flatten(), input_shape=input_shape))
model.add(LSTM(128, return_sequences=False))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Step 6: Train the model
num_epochs = 50
model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test))
