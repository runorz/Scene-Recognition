# -*- coding: utf-8 -*-

#Use google colab and drive
# from google.colab import drive
# drive.mount('/content/drive')

#image: np.ndarray, only accept 2-D grayscale image with shape in 2 dimension
#step: how many pizels the patch move
#patch_size: the size of the patch
#return the array with all the features, with shape (feature_amount, patch_size * patch_size)
def extractFeaturesFromImage(image, step, patch_size):
  width = image.shape[0]
  height = image.shape[1]
  features = []
  for y in range(0, height, step):
    if height - y < patch_size:
      break
    for x in range(0, width, step):
      if width - x < patch_size:
        break
      features.append(np.concatenate(image[x: x+patch_size, y: y+patch_size]))
  return np.asarray(features)

#image_features: the features array from images, should be done zero-mean and unit variace with the features from extractFeaturesFromImage()
#trained_KMeans: the KMeans model that has been trained, which will return the centers each feature belongs to, while input the features
def quantisation(image_features, trained_KMeans):
  n_centers = trained_KMeans.cluster_centers_.shape[0]
  visual_words_hist = np.zeros(n_centers)
  predictions = trained_KMeans.predict(image_features)
  for index in predictions:
    visual_words_hist[index] = visual_words_hist[index] + 1
  return visual_words_hist

import numpy as np
import os
import cv2
from sklearn import preprocessing


#The directory where training and testing directory were set
directory = "/content/drive/My Drive/"

train_directory = directory + "training/"
test_directory = directory + "testing/"

#All the class lebel, this is a List variable
labels = os.listdir(train_directory)

step = 4

patch_size = 8

train_images_features = []

train_target = []

#Load all images but only store their features on a list
for index in range(len(labels)):
  for name in os.listdir(train_directory + labels[index] + "/"):
    train_images_features.append(extractFeaturesFromImage(cv2.imread(train_directory + labels[index] + "/" + name, cv2.IMREAD_GRAYSCALE), step, patch_size))
    train_target.append(index)

#It takes 5-10 minutes to run KMeans for k = 1000
k = 1000

from sklearn import preprocessing
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans

#Do zero-mean and unit variace to the all training features array
scaled_train_images_features = preprocessing.scale(np.concatenate(train_images_features))

#Record the parameter used for cluster scaling, then use it to scale test data
scaler_for_clustering = preprocessing.StandardScaler().fit(np.concatenate(train_images_features))

#KMeans clustering
kmeans = MiniBatchKMeans(k, init_size=3*k).fit(scaled_train_images_features)

train_visual_word_hists = []

#quntisation every image
for one_image_features in train_images_features:
  train_visual_word_hists.append(quantisation(scaler_for_clustering.transform(one_image_features), kmeans))

#Do zero-mean and unit variance to the visual word histogram
scaled_train_visual_word_hists = preprocessing.scale(np.asarray(train_visual_word_hists))

#Record the parameter used for classification scaling, then use it to scale test data
scaler_for_classification = preprocessing.StandardScaler().fit(np.asarray(train_visual_word_hists))

scaled_train_visual_word_hists.shape

from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier

from sklearn.svm import LinearSVC

clf = OneVsOneClassifier(LinearSVC(max_iter=20000)).fit(scaled_train_visual_word_hists, np.asarray(train_target))

f = open("run2.txt", "w+")

#Used trained classifier to predict the class of images in testset and write it into text file
#This takes longer time than loading training image, about 5-10 minutes
for name in os.listdir(test_directory):
  test_image = cv2.imread(test_directory+name, cv2.IMREAD_GRAYSCALE)
  test_image_features = extractFeaturesFromImage(test_image, step, patch_size)
  scaled_test_image_features = scaler_for_clustering.transform(test_image_features)
  test_image_hist = quantisation(scaled_test_image_features, kmeans)
  test_image_hist = test_image_hist.reshape(1,k)
  scaled_test_image_hist = scaler_for_classification.transform(test_image_hist)
  predicted_class = clf.predict(scaled_test_image_hist)
  f.write(name+" "+labels[int(predicted_class[0])]+"\n")

f.close()
