# Detecção de Anomalias em Imagens usando VGG16 e Grad-CAM

Este é um programa Python que usa a arquitetura VGG16 pré-treinada para realizar a detecção de anomalias em imagens. Ele também utiliza a técnica Grad-CAM para visualizar mapas de ativação que destacam as áreas da imagem que mais influenciaram nas previsões.

## Requisitos

Certifique-se de que você tenha instalado as seguintes bibliotecas Python:

- OpenCV (cv2)
- NumPy (numpy)
- Imutils (imutils)
- Matplotlib (matplotlib)
- Keras (keras)
- TensorFlow (tensorflow)
- Scikit-learn (sklearn)

Você pode instalá-las usando pip:

pip install opencv-python-headless numpy imutils matplotlib


## Arquitetura do Modelo

O modelo personalizado utiliza a arquitetura VGG16 como base e adiciona camadas de classificação personalizadas no topo. Ele é treinado para classificar imagens em duas classes: "good" e "anomaly".

## Visualização Grad-CAM

O programa usa a técnica Grad-CAM para gerar mapas de ativação que mostram as áreas de uma imagem que mais influenciaram nas previsões do modelo. Isso ajuda na interpretação das previsões e na identificação de anomalias.
