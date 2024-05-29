# Face_Emotion_Detection
![image](https://user-images.githubusercontent.com/66847170/140658286-597a6944-2147-4197-ba18-6a846d980638.png)

# Introduction
This project aims to classify the emotion on a person's face into one of seven categories, using deep convolutional neural networks. The model is trained on the FER-2013 dataset which was published on International Conference on Machine Learning (ICML). This dataset consists of 35887 grayscale, 48x48 sized face images with seven emotions - angry, disgusted, fearful, happy, neutral, sad and surprised.

# Dataset
Fro this project we have kaggle dataset fer 2013: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data.

dataset_shape=(35887,3)

After reshaping for data:-

train shape (28709, 48, 48, 1) validation shape (3589, 48, 48, 1) validation shape (3589, 48, 48, 1)

# Face Emotion Recognition Model
Facial expression recognition system is a computer-based technology and therefore, it uses algorithms to instantaneously detect faces, code facial expressions, and recognize emotional states. It does this by analyzing faces in images or video through computer powered cameras embedded in laptops, mobile phones, and digital signage systems, or cameras that are mounted onto computer screens. Facial analysis through computer powered cameras generally follows three steps:

# A. Face detection

Locating faces in the scene, in an image or video footage.

# B. Facial Feature Detection

Extracting information about facial features from detected faces. For example, detecting the shape of facial components or describing the texture of the skin in a facial area.

# C. Facial expression and emotion Classification

Analyzing the movement of facial features and/or changes in the appearance of facial features and classifying this information into expression-interpretative categories such as facial muscle activations like smile or frown; emotion categories happiness or anger; attitude categories like (dis)liking or ambivalence

#  Face Detection
Face detection can be regarded as a specific case of object-class detection. In object-class detection, the task is to find the locations and sizes of all objects in an image that belong to a given class. Examples include upper torsos, pedestrians, and cars. Face detection simply answers two question, 1. are there any human faces in the collected images or video? 2. where is the located?

Face-detection algorithms focus on the detection of frontal human faces. It is analogous to image detection in which the image of a person is matched bit by bit. Image matches with the image stores in database. Any facial feature changes in the database will invalidate the matching process.

Object Detection using Haar feature-based cascade classifiers is an effective object detection method proposed by Paul Viola and Michael Jones in their paper, "Rapid Object Detection using a Boosted Cascade of Simple Features" in 2001. It is a machine learning based approach where a cascade function is trained from a lot of positive and negative images. It is then used to detect objects in other images. Here it will train with faces. Initially, the algorithm needs a lot of positive images (images of faces) and negative images (images without faces) to train the classifier.

#  Facial Feature Detection and Emotion Classification.
The Haar Casscade detects face and those faces are then cropped and convert to gray images. These gray images further get converted into iamge aaray for processing. 

![image](https://user-images.githubusercontent.com/66847170/140658525-a3be16ed-3438-4632-b30d-c65abd47638d.png)

# Experiment With Model

# 1.Using MLP with tensorflow version 2.4.1
Epoch 1/5 898/898 [==============================] - 4s 4ms/step - loss: 47.1633 - accuracy: 0.2087 - val_loss: 2.3757 - val_accuracy: 0.2282
Epoch 2/5 898/898 [==============================] - 3s 4ms/step - loss: 1.9894 - accuracy: 0.2493 - val_loss: 2.1619 - val_accuracy: 0.1828

Epoch 3/5 898/898 [==============================] - 3s 4ms/step - loss: 1.8900 - accuracy: 0.2655 - val_loss: 2.6126 - val_accuracy: 0.1167

Epoch 4/5 898/898 [==============================] - 3s 4ms/step - loss: 1.8567 - accuracy: 0.2327 - val_loss: 1.8168 - val_accuracy: 0.2449

Epoch 5/5 898/898 [==============================] - 3s 4ms/step - loss: 1.8110 - accuracy: 0.2486 - val_loss: 1.8177 - val_accuracy: 0.2449 <tensorflow.python.keras.callbacks.History at 0x7fd02058f390>

# 2.Using CNN
Epoch 1/15 684/684 [==============================] - 56s 82ms/step - loss: 0.6443 - accuracy: 0.7727 - val_loss: 3.1014 - val_accuracy: 0.4260

Epoch 2/15 684/684 [==============================] - 56s 81ms/step - loss: 0.5620 - accuracy: 0.8021 - val_loss: 3.4424 - val_accuracy: 0.3954

Epoch 3/15 684/684 [==============================] - 56s 81ms/step - loss: 0.5157 - accuracy: 0.8240 - val_loss: 3.6780 - val_accuracy: 0.4124

Epoch 4/15 684/684 [==============================] - 56s 82ms/step - loss: 0.4715 - accuracy: 0.8365 - val_loss: 4.1690 - val_accuracy: 0.4154

Epoch 5/15 684/684 [==============================] - 56s 82ms/step - loss: 0.4663 - accuracy: 0.8418 - val_loss: 4.1743 - val_accuracy: 0.4118

Epoch 6/15 684/684 [==============================] - 56s 82ms/step - loss: 0.4277 - accuracy: 0.8535 - val_loss: 4.6127 - val_accuracy: 0.3996

Epoch 7/15 684/684 [==============================] - 56s 82ms/step - loss: 0.3749 - accuracy: 0.8723 - val_loss: 4.8611 - val_accuracy: 0.4093

Epoch 8/15 684/684 [==============================] - 56s 82ms/step - loss: 0.3612 - accuracy: 0.8795 - val_loss: 5.1365 - val_accuracy: 0.3929

Epoch 9/15 684/684 [==============================] - 56s 82ms/step - loss: 0.3314 - accuracy: 0.8880 - val_loss: 5.3357 - val_accuracy: 0.4018

Epoch 10/15 684/684 [==============================] - 56s 81ms/step - loss: 0.3101 - accuracy: 0.8975 - val_loss: 5.6454 - val_accuracy: 0.4163

Epoch 11/15 684/684 [==============================] - 56s 82ms/step - loss: 0.2928 - accuracy: 0.9048 - val_loss: 5.8554 - val_accuracy: 0.4082

Epoch 12/15 684/684 [==============================] - 56s 82ms/step - loss: 0.3159 - accuracy: 0.8982 - val_loss: 6.0766 - val_accuracy: 0.4104

Epoch 13/15 684/684 [==============================] - 56s 82ms/step - loss: 0.2567 - accuracy: 0.9161 - val_loss: 6.5591 - val_accuracy: 0.4132

Epoch 14/15 684/684 [==============================] - 56s 82ms/step - loss: 0.2475 - accuracy: 0.9221 - val_loss: 6.6915 - val_accuracy: 0.4082

Epoch 15/15 684/684 [==============================] - 56s 82ms/step - loss: 0.2542 - accuracy: 0.9191 - val_loss: 7.1066 - val_accuracy: 0.4065

<tensorflow.python.keras.callbacks.History at 0x7fd01a0b55
