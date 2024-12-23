# Next Frame Video Prediction with Convolutional LSTMs
 In this project, 5 walking animations, each consisting of approximately 250 frames and lasting about 10 seconds, created with Blender 3.5 animation software were used. Convolutional LSTMs (Convolutional LSTMs) architecture was utilized for processing and predicting the images. Each animation video in the dataset was converted to grayscale, resized to 64x64, and combined into moving npy files consisting of 24 frames. As a result, a single-channel dataset of 1250 sequences, each 24 frames long and 64x64 in size, was created. The predictions generated at the end of the study were presented as GIFs.

 The processes of converting the animations to grayscale, resizing them to 64x64, and transforming them into 24-frame long moving npy files were carried out in the data_preprocessing.py file. (All animations used were individually subjected to this conversion process.)

 

https://github.com/user-attachments/assets/02aa4a01-8e65-4182-8d8f-642216900126


![predicted_video_2](https://github.com/user-attachments/assets/832bc364-c596-4b0b-896b-af5d3c875268)   Real frames
![predicted_video_3](https://github.com/user-attachments/assets/6fd343eb-a199-4ec0-a085-0c2d2dedde3c)   Predicted frames
