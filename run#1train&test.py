# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2

from sklearn import neighbors

#k for K-Nearest-Neighbour
n_neighbors = 3

#The size all image will be resized to
tiny_size = 16

#The directory where training and testing directory were set.
#If data was set in current directory, change it to empty string
directory = "/content/drive/My Drive/"

training_directory = directory + 'training/'
test_directory = directory + 'testing/'

labels = os.listdir(training_directory)

training_images = []
training_classes = []

#Load all the training image, and their corresponding class index, it may takes couple of minutes
for index in range(len(labels)):
  for name in os.listdir(training_directory+labels[index]):
    training_images.append(np.concatenate(cv2.resize(cv2.imread(training_directory+labels[index]+"/"+name, cv2.IMREAD_GRAYSCALE), (tiny_size, tiny_size))))
    training_classes.append(index)

clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')

#Train the classifier
clf.fit(np.asarray(training_images), np.asarray(training_classes))

f = open("run1.txt", "w+")

#Used trained classifier to predict the class of images in testset and write it into text file
#This takes longer time than loading training image, about 5-10 minutes
for name in os.listdir(test_directory):
  predicted_class = clf.predict(np.concatenate(cv2.resize(cv2.imread(test_directory+name, cv2.IMREAD_GRAYSCALE), (tiny_size, tiny_size))).reshape((1, tiny_size*tiny_size)))
  f.write(name+" "+labels[int(predicted_class[0])]+"\n")

f.close()

