"""
"Código para classificação de anomalias rede neural simples"
"""
import os
import random
import cv2
import numpy as np
from imutils import paths
from keras.layers import Dense
from keras.models import Sequential
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.dummy import DummyClassifier


# Dataset path
dataset_path = "/home/iapfa/code/automotive_radiator/Data"
SEED=20


print("[INFO] Loading images...")
data = []
labels = []
data_final = []
labels_final = []

# Get image paths and shuffle them randomly
image_paths = sorted(list(paths.list_images(dataset_path)))
random.seed(SEED)
random.shuffle(image_paths)



# Load images and labels
for image_path in image_paths:
    image = cv2.imread(image_path)
    image = cv2.resize(image, (64, 64)).flatten()
    data.append(image)

    label = image_path.split(os.path.sep)[-2]
    labels.append(label)

    final_test = cv2.flip(image, 1).flatten()
    data_final.append(final_test)
    labels_final.append(label)

# Convert to numpy arrays and normalize pixel values
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

data_final = np.array(data_final, dtype="float") / 255.0
labels_final = np.array(labels_final)

# Split into training and testing sets
(train_x, test_x, train_y, test_y) = train_test_split(data, labels, test_size=0.25, random_state=SEED, stratify=labels)

# Encode labels
lb = LabelBinarizer()
train_y = lb.fit_transform(train_y)
test_y = lb.transform(test_y)
y_final = lb.transform(labels_final)

# Create neural network model
model = Sequential()
model.add(Dense(1024, input_shape=(data.shape[1],), activation="sigmoid"))
model.add(Dense(512, activation="sigmoid"))
model.add(Dense(1, activation="sigmoid"))

init_lr = 0.01
epochs = 80

# Compile the model
print("[INFO] Training neural network...")
model.compile(loss="binary_crossentropy",
              optimizer="adam", metrics=["accuracy"])

# Train the model
history = model.fit(x=train_x, y=train_y, validation_data=(test_x, test_y),
                    epochs=epochs, batch_size=32)



# Evaluate the model on the test set
print("[INFO] Evaluating neural network on test set...")
predictions = model.predict(x=test_x, batch_size=32)
threshold = 0.5
predictions_binary = (predictions > threshold).astype(int)
accuracy_model = accuracy_score(test_y, predictions_binary)
print("Test set accuracy:", accuracy_model)
classification_rep = classification_report(
    test_y,
    predictions_binary,
    target_names=lb.classes_
)
print(classification_rep)


dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(train_x, train_y)
dummy_predictions = dummy_clf.predict(test_x)
dummy_accuracy = accuracy_score(test_y, dummy_predictions)
print("Dummy Classifier accuracy:", dummy_accuracy)
dummy_classification_rep = classification_report(
    test_y, 
    dummy_predictions, 
    target_names=lb.classes_
)
print(dummy_classification_rep)

# Evaluate the model on the final dataset
print("[INFO] Evaluating neural network on final dataset...")
predictions = model.predict(x=data_final, batch_size=32)
predictions_binary = (predictions > threshold).astype(int)
accuracy_model = accuracy_score(y_final, predictions_binary)
print("Final dataset accuracy:", accuracy_model)
classification_rep = classification_report(
    y_final,
    predictions_binary,
    target_names=lb.classes_
)
print(classification_rep)
