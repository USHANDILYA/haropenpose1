# import cv2
# import numpy as np
# from tf_pose import common
# from tf_pose.estimator import TfPoseEstimator
# from tf_pose.networks import get_graph_path

# # Set the path to the graph file
# model = get_graph_path('mobilenet_thin')

# # Initialize TfPoseEstimator
# tfpose_estimator = TfPoseEstimator(model, target_size=(368, 432))

# def get_keypoints(video_path):
#     cap = cv2.VideoCapture(video_path)
#     keypoints = []

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Resize frame to fit the network input size
#         resized_frame = tfpose_estimator.resize_and_pad_image(frame)

#         # Perform pose estimation
#         humans = tfpose_estimator.inference(resized_frame, resize_to_default=True, upsample_size=4.0)

#         if len(humans) > 0:
#             # Extract keypoints for the first detected human (assuming single person)
#             human = humans[0]
#             keypoints.append(np.array([(human.body_parts[i].x, human.body_parts[i].y, human.body_parts[i].score)
#                                        if i in human.body_parts.keys() else (0, 0, 0) for i in range(common.CocoPart.Background.value)]))
#         else:
#             keypoints.append(np.zeros((common.CocoPart.Background.value, 3)))  # Return zeros if no keypoints detected

#     cap.release()
#     return np.array(keypoints)

# # # Example usage
# # video_path = "path_to_video.mp4"
# # keypoints = get_keypoints(video_path)
# np.save("keypoints.npy", keypoints)  # Save keypoints for later use
import cv2
import numpy as np
from tf_pose import common
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path
import os
# Set the path to the graph file
model = get_graph_path('mobilenet_thin')
# graph_path = os.path.join(base_data_dir, dyn_graph_path[model_name])
# Initialize TfPoseEstimator
tfpose_estimator = TfPoseEstimator(model, target_size=(368, 432))

def get_keypoints(video_path, batch_size=1):
    """
    Extracts keypoints from a video using TfPoseEstimator.

    Args:
        video_path (str): Path to the video file.
        batch_size (int, optional): The number of frames to process together for efficiency. Defaults to 1.

    Returns:
        np.array: A numpy array containing keypoints for each frame in the video.
    """

    cap = cv2.VideoCapture(video_path)
    keyframes = []  # List to store keypoints for each batch (or frame)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frames in batches for efficiency (optional)
        frames = []
        while len(frames) < batch_size and ret:
            frames.append(frame)
            ret, frame = cap.read()

        if len(frames) > 0:
            # Resize and pad the frames to match the expected input shape
            resized_frames = [tfpose_estimator.resize_and_pad_image(f) for f in frames]

            # Perform pose estimation on the batch of frames
            humans_batch = tfpose_estimator.inference(resized_frames, resize_to_default=True, upsample_size=4.0)

            for i, humans in enumerate(humans_batch):
                if len(humans) > 0:
                    # Extract keypoints for the first detected human (assuming single person)
                    human = humans[0]
                    keypoints = np.array([(human.body_parts[i].x, human.body_parts[i].y, human.body_parts[i].score)
                                          if i in human.body_parts.keys() else (0, 0, 0) for i in range(common.CocoPart.Background.value)])
                else:
                    # Return zeros if no keypoints detected
                    keypoints = np.zeros((common.CocoPart.Background.value, 3))

                # Store keypoints based on batch processing or individual frame
                if batch_size == 1:
                    keyframes.append(keypoints)
                else:
                    keyframes.append(keypoints[:i+1])  # Append keypoints for processed frames in the batch

    cap.release()

    # Specify the full path where you want to save the keypoints file
    #keypoints_file_path = r'C:\Users\UTKARSH\Desktop\data science\dl\har2\keypoints.npy'

# Save keypoints to file
    #np.save(keypoints_file_path, np.array(keyframes))
    np.save('keypoints.npy',np.array(keyframes) )
    return np.array(keyframes)

    # # Save keypoints to file
    
    




