import cv2
import glob
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Change the dataset_path to point to the unzipped Dataset_1/images folder in your computer.
dataset_path = "Dataset/CTSD-test/images/"

X = []
y = []
for i in glob.glob(dataset_path + '*.png', recursive=True):
    label = i.split("images")[1][1:4]
    y.append(label)
    # read the images and append them to list X
    X.append(cv2.imread(i))

X_processed = []
for x in X:
    # resize them to 48x48 pixels.
    temp_x = cv2.resize(x, (48, 48))
    # convert the images to grayscale using the cvtColor function in opencv
    temp_x = cv2.cvtColor(temp_x, cv2.COLOR_BGR2GRAY)
    # append the pre-processed images to X_processed list.
    X_processed.append(temp_x)

X_features = []
for x in X_processed:
    # extract hog features
    x_feature = hog(x, orientations=8, pixels_per_cell=(10, 10),
                    cells_per_block=(1, 1), visualize=False)
    X_features.append(x_feature)

# split X_features and y into training and testing sets
# Use a 80-20 split and make sure to shuffle the samples.
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42, shuffle=True)

# Use the sklearn SVM package to train a classifier using x_train and y_train.
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Use the x_test and y_test to evaluate the classifier and print the accuracy value.
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")