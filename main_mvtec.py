import os
import random
import cv2
import numpy as np
from imutils import paths

import matplotlib.pyplot as plt

from keras.applications import VGG16
from keras.layers import Input, MaxPooling2D, AveragePooling2D, Flatten, Dense,Conv2D, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Sequential

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.dummy import DummyClassifier
from sklearn.utils.class_weight import compute_class_weight

def create_custom_vgg(n_classes=2):
    # Carregue o modelo VGG16 pré-treinado sem a camada de classificação (include_top=False)
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Congele as camadas convolucionais do modelo base
    for layer in base_model.layers:
        layer.trainable = False
    
    # Crie a cabeça de classificação personalizada
    x = MaxPooling2D(pool_size=(2, 2))(base_model.output)
    x = AveragePooling2D(pool_size=(3, 3))(x)
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    output = Dense(n_classes, activation='softmax')(x)
    
    
    # Crie o modelo final
    custom_model = Model(inputs=base_model.input, outputs=output)



    
    return custom_model


def plot_metrics(history):
  metrics = ['loss', 'accuracy']
  for n, metric in enumerate(metrics):
    name = metric.replace("_"," ").capitalize()
    plt.subplot(2,2,n+1)
    plt.plot(history.epoch, history.history[metric], color='blue', label='Train')
    plt.plot(history.epoch, history.history['val_'+metric],
             color='red', linestyle="--", label='Val')
    plt.xlabel('Epoch')
    plt.ylabel(name)
    if metric == 'loss':
      plt.ylim([0, plt.ylim()[1]])
    elif metric == 'auc':
      plt.ylim([0.8,1])
    else:
      plt.ylim([0,1])

    plt.legend();


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()



dataset_path = "/home/iapfa/code/automotive_radiator/Data/mvtec_anomaly_detection/pill/images"
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

epochs=10



# Load images and labels
for image_path in image_paths:
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    data.append(image)
    label = image_path.split(os.path.sep)[-2]
    if label == 'good':
        labels.append([1,0])
    else:
        labels.append([0,1])


data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

(train_x, test_x, train_y, test_y) = train_test_split(data, labels, test_size=0.2, random_state=SEED, stratify=labels)


lb = LabelBinarizer()
train_y = lb.fit_transform(train_y)
test_y = lb.transform(test_y)


batch_size=10
custom_optimizer = Adam(lr=0.0001)
# Crie o modelo personalizado com 2 classes de saída
model = create_custom_vgg(n_classes=2)

# Compile o modelo com otimizador, função de perda e métricas adequados
model.compile(optimizer=custom_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Visualize a arquitetura do modelo
model.summary()

## Repensar o  class_weight
# class_weight = train_y[:,1]
# class_weights = compute_class_weight('balanced', classes=np.unique(class_weight), y=class_weight)
# class_weight_dict = dict(enumerate(class_weights))



history = model.fit(x=train_x, y=train_y, validation_data=(test_x, test_y),
                    epochs=epochs, batch_size=batch_size)

print("[INFO] Evaluating neural network on test set...")
predictions = model.predict(x=test_x, batch_size=batch_size)
threshold = 0.5
predictions_binary = (predictions > threshold).astype(int)
accuracy_model = accuracy_score(test_y, predictions_binary)
print("Test set accuracy:", accuracy_model)


#Analise resultados
layer_name = 'block5_conv3' 
for image,label,result in zip(test_x,test_y,predictions):
    if label[1]==1 or result[1]>0.5:
        print (label, result)
        img_tensor = np.expand_dims(image, axis=0)

        heatmap = make_gradcam_heatmap(img_tensor, model, layer_name,1)
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(np.squeeze(img_tensor))
        plt.title('Imagem Original')

        plt.subplot(1, 2, 2)
        plt.imshow(heatmap)
        plt.title('Heatmap de Ativação')



        plt.colorbar()
        plt.show()
        input("Próximo")
