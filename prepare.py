import numpy as np
import tensorflow as tf
#keypoints = np.load("keypoints.npy", allow_pickle=True)  # Load keypoints from file

class KeypointsDataset(tf.data.Dataset):
    def _generator(keypoints, labels):
        for keypoint, label in zip(keypoints, labels):
            yield keypoint, label

    def __new__(cls, keypoints, labels):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_signature=(
                tf.TensorSpec(shape=(None, 25, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int64)
            ),
            args=(keypoints, labels)
        )

# Preprocessing function for keypoints
def preprocess_keypoints(keypoint, label):
    # Normalize keypoints
    keypoint = (keypoint - np.mean(keypoint, axis=0)) / np.std(keypoint, axis=0)
    
    # Pad keypoints to ensure a fixed length
    max_length = 100  # Define your maximum sequence length
    if len(keypoint) < max_length:
        pad_length = max_length - len(keypoint)
        keypoint = np.pad(keypoint, ((0, pad_length), (0, 0), (0, 0)), 'constant')
    else:
        keypoint = keypoint[:max_length, :, :]  # Trim keypoints if longer than max_length
    
    return keypoint, label

# Load keypoints and create dataset
keypoints = np.load("keypoints.npy")  # Load keypoints from file
labels = ...  # Define your labels for each video/frame
dataset = KeypointsDataset(keypoints, labels)

# Apply preprocessing and prepare dataset for LSTM input
dataset = dataset.map(preprocess_keypoints)
dataset = dataset.batch(32).shuffle(buffer_size=100)
# import tensorflow as tf

# class KeypointsDataset(tf.data.Dataset):
#     def _generator(keypoints, labels):
#         for keypoint, label in zip(keypoints, labels):
#             yield keypoint, label

#     def __new__(cls, keypoints, labels):
#         return tf.data.Dataset.from_generator(
#             cls._generator,
#             output_signature=(
#                 tf.TensorSpec(shape=(None, 25, 3), dtype=tf.float32),
#                 tf.TensorSpec(shape=(), dtype=tf.int64)
#             ),
#             args=(keypoints, labels)
#         )

# # Preprocessing function for keypoints
# def preprocess_keypoints(keypoint, label):
#     # Normalize keypoints
#     keypoint = (keypoint - np.mean(keypoint, axis=0)) / np.std(keypoint, axis=0)
    
#     # Pad keypoints to ensure a fixed length
#     max_length = 100  # Define your maximum sequence length
#     if len(keypoint) < max_length:
#         pad_length = max_length - len(keypoint)
#         keypoint = np.pad(keypoint, ((0, pad_length), (0, 0), (0, 0)), 'constant')
#     else:
#         keypoint = keypoint[:max_length, :, :]  # Trim keypoints if longer than max_length
    
#     return keypoint, label

# # Load keypoints and create dataset
# keypoints = np.load("keypoints.npy")  # Load keypoints from file

# # Define your labels for each video/frame
# labels = ...  

# # Create dataset
# dataset = KeypointsDataset(keypoints, labels)

# # Apply preprocessing and prepare dataset for LSTM input
# dataset = dataset.map(preprocess_keypoints)
# dataset = dataset.batch(32).shuffle(buffer_size=100)
