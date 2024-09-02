# Deep-Learning-for-Word-Level-American-Sign-Language-WLASL-translation

In this group work, we use various Deep Learning models in order to perform video classification for sign language dataset: World-Level American Sign Language (WLASL)[1].
The model I focused on was a Convolutional Neural Network (CNN), which translated a video in a 3X3 image and performed image classification on this new dataset. The 3X3 images were created by extracting 9 frames, in the middle of the video, and form a 3X3 square grid. Additionally, there were applied data augmentation techniques, specifically: zoom, speedup and horizontal flip. Finally, I created a CNN with 2 layers composed of Convolution, Batch Normalization (the batch were formed of 10 images) and Average Pooling, this is followed by a dense Multi Layer Perceptron (MLP) that will finally classifiy the images.

# Bibliography
[1] https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed
