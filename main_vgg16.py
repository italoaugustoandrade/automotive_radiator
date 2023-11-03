"Código para classificação de anomalias rede neural VGG116"

import os
import random
import cv2
import numpy as np
from imutils import paths
import matplotlib.pyplot as plt
import tensorflow as tf


from keras.applications import VGG16
from keras.layers import Input, MaxPooling2D, AveragePooling2D, Flatten, Dense, Conv2D
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

# Função para criar um modelo personalizado usando a arquitetura VGG16
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

# Função para plotar métricas de treinamento
def plot_metrics(history):
    metrics = ['loss', 'accuracy']
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(2, 2, n+1)
        plt.plot(history.epoch, history.history[metric], color='blue', label='Train')
        plt.plot(history.epoch, history.history['val_'+metric], color='red', linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'accuracy':
            plt.ylim([0, 1])
        plt.legend()

# Função para gerar mapas de ativação Grad-CAM
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Crie um modelo que mapeia a imagem de entrada para as ativações da última camada convolucional e as previsões de saída
    grad_model = Model(model.inputs, [model.get_layer(last_conv_layer_name).output, model.output])

    # Calcule o gradiente da classe predita (ou escolhida) em relação às ativações da última camada convolucional
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Este é o gradiente da saída (classe superior prevista ou escolhida)
    # com relação ao mapa de características de saída da última camada convolucional
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Este é um vetor onde cada entrada é a intensidade média do gradiente sobre um canal de mapa de características específico
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiplique cada canal no array do mapa de características
    # pela "importância deste canal" em relação à classe prevista superior
    # e depois some todos os canais para obter o mapa de ativação da classe
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Para fins de visualização, normalize o mapa de ativação entre 0 e 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Diretório do conjunto de dados de imagens
dataset_path = "/home/iapfa/code/automotive_radiator/Data/mvtec_anomaly_detection/pill/images"
SEED = 20

# Carregamento de imagens e pré-processamento
print("[INFO] Loading images...")
data = []
labels = []

# Get image paths and shuffle them randomly
image_paths = sorted(list(paths.list_images(dataset_path)))
random.seed(SEED)
random.shuffle(image_paths)

epochs = 10

# Carregue imagens e rótulos
for image_path in image_paths:
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    data.append(image)
    label = image_path.split(os.path.sep)[-2]
    if label == 'good':
        labels.append([1, 0])
    else:
        labels.append([0, 1])

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# Divisão dos dados em conjuntos de treinamento e teste
(train_x, test_x, train_y, test_y) = train_test_split(data, labels, test_size=0.2, random_state=SEED, stratify=labels)

# Conversão de rótulos em codificação one-hot
lb = LabelBinarizer()
train_y = lb.fit_transform(train_y)
test_y = lb.transform(test_y)

# Tamanho do lote para treinamento
batch_size = 10

# Criação do otimizador personalizado (Adam)
custom_optimizer = Adam(lr=0.0001)

# Criação do modelo personalizado com 2 classes de saída
model = create_custom_vgg(n_classes=2)

# Compilação do modelo com otimizador, função de perda e métricas adequados
model.compile(optimizer=custom_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Resumo da arquitetura do modelo
model.summary()

# Treinamento do modelo
history = model.fit(x=train_x, y=train_y, validation_data=(test_x, test_y), epochs=epochs, batch_size=batch_size)

# Avaliação do modelo no conjunto de teste
print("[INFO] Evaluating neural network on test set...")
predictions = model.predict(x=test_x, batch_size=batch_size)
threshold = 0.5
predictions_binary = (predictions > threshold).astype(int)
accuracy_model = accuracy_score(test_y, predictions_binary)
print("Test set accuracy:", accuracy_model)

# Visualização dos mapas de ativação Grad-CAM para imagens de teste
layer_name = 'block5_conv3' 
for image, label, result in zip(test_x, test_y, predictions):
    if label[1] == 1 or result[1] > 0.5:
        print(label, result)
        img_tensor = np.expand_dims(image, axis=0)

        heatmap = make_gradcam_heatmap(img_tensor, model, layer_name, 1)
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
