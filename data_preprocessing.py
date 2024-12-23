import cv2
import numpy as np

def resize_frame(frame, target_size):
    resized_frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
    return resized_frame

def convert_to_grayscale(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray_frame

video_path = "walk_victoria0001-0250.mp4"
cap = cv2.VideoCapture(video_path)

target_size = (64,64)
n_frames = 24


moving_snow = np.zeros((n_frames, 250, *target_size), dtype=np.uint8)

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    resized_frame = resize_frame(frame, target_size)
    grayscale_frame = convert_to_grayscale(resized_frame)
    
    time_step = frame_count % n_frames
    moving_snow[time_step] = grayscale_frame
    
    frame_count += 1

cap.release()
np.save("moving_victoria_24.npy", moving_snow)


data = np.load("moving_victoria_24.npy")
shape = data.shape
print("Size of the Numpy array:", shape)

npy_files = ["moving_jay_24.npy",
             "moving_phil_24.npy", 
             "moving_rain_24.npy", 
             "moving_snow_24.npy", 
             "moving_victoria_24.npy"]
arrays = []
for file in npy_files:
    data = np.load(file)
    arrays.append(data)  
merged_array = np.concatenate(arrays, axis=1)  

np.save("merged_24.npy", merged_array)



data = np.load("merged_24.npy")
shape = data.shape
print("Size of the Numpy array:", shape)