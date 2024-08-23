# Emotion_detection
The notebook uploaded consists of code used to train the model for detecting emotions. The dataset used for training the model can found at this [link](https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer).

The model used for training consists of consists of four convolutional layers with increasing filter sizes, each followed by batch normalization, ReLU activation, max pooling, and dropout for regularization. 

The network then has two fully connected dense layers before the final softmax layer, using the Adam optimizer and categorical cross-entropy loss for training. 

Such networks are suitable for emotion detection. 

The following GIF shows the result of the trained model.



<img src="https://github.com/user-attachments/assets/435790fd-63dc-49c4-91a0-6fea064ad6d5" alt="Emotion_detection" width="400">

