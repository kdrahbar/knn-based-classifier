from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn import decomposition
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as pl
import Image

# Make all images the same size
IMG_SIZE = (300, 167)

# Abstract only 2 features from the images and make a 2-d graph out of it
def graph(data, y):
	pca = decomposition.RandomizedPCA(n_components=2)
	X = np.array(pca.fit_transform(data))
	df = pd.DataFrame({"x": X[:, 0], "y": X[:, 1], "label":np.where(y==1, "Check", "Driver's License")})
	colors = ["red", "yellow"]
	for label, color in zip(df['label'].unique(), colors):
	    mask = df['label']==label
	    pl.scatter(df[mask]['x'], df[mask]['y'], c=color, label=label)
	pl.legend()
	pl.show()

# Make a 2d numpy array of the raw image
def make_np_array(imageName):
	print(imageName)
	img = Image.open(imageName)
	print('\n')
	img = img.resize(IMG_SIZE)
	img = np.array(img)
	return img

# Resizes the image and then turns it into a 1d numpy array
def serialize(img):
    tup = img.shape
    x = tup[0]
    y = tup[1]
    s = x * y
    shortened = [[0 for x in xrange(len(img[0]))] for x in xrange(len(img))]
    if(len(img.shape) == 3):
    	columns = len(img)
    	rows = len(img[0])
    	for c in xrange(columns):
    		for r in xrange(rows):
    			shortened[c][r] = sum(img[c][r])/3
    	img = np.array(shortened)
    image_wide = img.reshape(1, s)
    print image_wide[0]
    return image_wide[0]

# Search the directory containing the images for the two classifier labels
img_dir = "images/"
images = [img_dir+ f for f in os.listdir(img_dir)]
labels = ["check" if "check" in f.split('/')[-1] else "drivers_license" for f in images]

data = []
for image in images:
    img = make_np_array(image)
    img = serialize(img)
    data.append(img)

# Create a bool array where true maps to a check image
labels = np.where(np.array(labels)=="check", 1, 0)

usr_input = raw_input("Do you want to see a 2d graph of the image? (y or n): ")
if(usr_input == 'y'):
	graph(data, labels)

# Randomly select ~70% of the data to be used in the training set
is_train = np.random.uniform(0, 1, len(data)) <= 0.7

data = np.array(data)
train_data, train_labels = data[is_train], labels[is_train]
test_data, test_labels = data[is_train==False], labels[is_train==False]

pca = decomposition.RandomizedPCA(n_components=5)
train_data = pca.fit_transform(train_data)
test_data = pca.fit_transform(test_data)

knn = KNeighborsClassifier()
knn.fit(train_data, train_labels)

print("Expected Results:"),
print(test_labels)
print("Actual Results:  "),
print(knn.predict(test_data))

results = [True for i, j in zip(test_labels, knn.predict(test_data)) if i == j]

print("The classifier identified"),
print(len(results)),
print("of the"),
print(len(test_labels)),
print("images")

