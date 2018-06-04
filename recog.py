from sklearn import datasets
from sklearn.svm import SVC
from scipy import misc
import numpy as np
import PIL

dig = datasets.load_digits()
features  = dig.data
numbers = dig.target

clf = SVC(gamma=0.00001)
clf.fit(features,numbers)

image = misc.imread('image.jpg')
image = misc.imresize(image,(8,8)).astype(dig.images.dtype)
image = misc.bytescale(image, high=16, low=0)


mat = []

for row in image:
	for col in row:
		mat.append(sum(col)/3.0)


print(clf.predict([mat]))