import os
import cv2
import mahotas  
import numpy as np
from sklearn import preprocessing
from progress.bar import Bar
import time

# Caminhos
trainImagePath = 'datasets_splitted/train/'
testImagePath = 'datasets_splitted/val/'
trainFeaturePath = 'labels/train/'
testFeaturePath = 'labels/val/'

def getData(path):
    images = []
    labels = []
    if os.path.exists(path):
        for dirpath, dirnames, filenames in os.walk(path):
            if len(filenames) > 0:  
                folder_name = os.path.basename(dirpath)  # covid ou normal
                bar = Bar(f'[INFO] Processando {folder_name}', max=len(filenames))
                for file in filenames:
                    label = folder_name
                    labels.append(label)
                    full_path = os.path.join(dirpath, file)
                    image = cv2.imread(full_path)
                    images.append(image)
                    bar.next()
                bar.finish()
        return images, np.array(labels)

def extractHaralickFeatures(images):
    featuresList = []
    bar = Bar('[INFO] Extraindo Features...', max=len(images))
    for image in images:
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Converte para escala de cinza
        blur_image = cv2.medianBlur(image, 3)  # Suavização
        features = mahotas.features.haralick(blur_image).mean(axis=0)  # Extrai Features Haralick
        featuresList.append(features)
        bar.next()
    bar.finish()
    return np.array(featuresList)

def encodeLabels(labels):
    encoder = preprocessing.LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)  # Converte labels para números
    return encoded_labels, encoder.classes_

def saveData(path, labels, features, encoderClasses):
    os.makedirs(path, exist_ok=True)
    np.savetxt(os.path.join(path, 'haralick_labels.csv'), labels, delimiter=',', fmt='%i')
    np.savetxt(os.path.join(path, 'haralick_features.csv'), features, delimiter=',')
    np.savetxt(os.path.join(path, 'haralick_classes.csv'), encoderClasses, delimiter=',', fmt='%s')

def main():
    print('[INFO] Processando Dados de Treino...')
    trainImages, trainLabels = getData(trainImagePath)
    trainEncodedLabels, encoderClasses = encodeLabels(trainLabels)
    trainFeatures = extractHaralickFeatures(trainImages)
    saveData(trainFeaturePath, trainEncodedLabels, trainFeatures, encoderClasses)

    print('[INFO] Processando Dados de Teste...')
    testImages, testLabels = getData(testImagePath)
    testEncodedLabels, _ = encodeLabels(testLabels)
    testFeatures = extractHaralickFeatures(testImages)
    saveData(testFeaturePath, testEncodedLabels, testFeatures, encoderClasses)

if __name__ == "__main__":
    main()
