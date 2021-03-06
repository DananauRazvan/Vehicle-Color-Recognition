import glob
import imageio
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.optimizers import Adam
import tensorflow
from imageai.Detection import ObjectDetection
from PIL import Image
from sklearn.decomposition import PCA
import cv2
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn_som.som import SOM
import os
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

def vehicleDetector(inputPath, outputPath):
    detector = ObjectDetection()

    modelPath = 'C:/Users/razva/OneDrive/Desktop/Vehicle Color Recognition/yolo.h5'

    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(modelPath, )
    detector.loadModel(detection_speed = 'flash')

    for idx, imagePath in enumerate(glob.glob(inputPath)):
        returnedImage, detection = detector.detectObjectsFromImage(input_image = imagePath, output_type = 'array')
        maxPercentageProbability = 0
        index = -1
        i = -1
        ok = False
        for eachItem in detection:
            i += 1
            if (eachItem['name'] == 'car' or eachItem['name'] == 'truck' or eachItem['name'] == 'motorcycle') and eachItem['percentage_probability'] > maxPercentageProbability:
                index = i
                maxPercentageProbability = eachItem['percentage_probability']
                ok = True

        if inputPath == 'C:/Users/razva/OneDrive/Desktop/Vehicle Color Recognition/dataset/train/*.jpg':
            imageType = 'Train image'
        else:
            imageType = 'Test image'

        if ok:
            print(imageType, str(idx) + ':', detection[index]['name'], detection[index]['percentage_probability'], detection[index]['box_points'])
            image = Image.open(imagePath)
            box = detection[index]['box_points']
            image = image.crop(box)
            image = image.resize((224, 224))
            image.save(outputPath + imagePath[71:])
        else:
            print(imageType, str(idx) + ':', 'undetectable')
            image = Image.open(imagePath)
            image = image.resize((224, 224))
            image.save(outputPath + imagePath[71:])
            continue

def principalComponentAnalysis(inputPath, outputPath):
    for idx, imagePath in enumerate(glob.glob(inputPath)):
        image = cv2.imread(imagePath)

        red, green, blue = cv2.split(image)

        dfRed = red / 255
        dfBlue = blue / 255
        dfGreen = green / 255

        pcaRed = PCA(n_components = 10)
        pcaRed.fit(dfRed)
        transPcaRed = pcaRed.transform(dfRed)
        pcaBlue = PCA(n_components = 10)
        pcaBlue.fit(dfBlue)
        transPcaBlue = pcaBlue.transform(dfBlue)
        pcaGreen = PCA(n_components = 10)
        pcaGreen.fit(dfGreen)
        transPcaGreen = pcaGreen.transform(dfGreen)

        redArr = pcaRed.inverse_transform(transPcaRed)
        blueArr = pcaBlue.inverse_transform(transPcaBlue)
        greenArr = pcaGreen.inverse_transform(transPcaGreen)

        imageReduced = (cv2.merge((redArr, greenArr, blueArr)))

        if inputPath == 'C:/Users/razva/OneDrive/Desktop/Vehicle Color Recognition/dataset/newTrain/*.jpg':
            name = imagePath[75:]
            imageType = 'Train image'
        else:
            name = imagePath[74:]
            imageType = 'Test image'

        cv2.imwrite(outputPath + name, imageReduced * 255)
        print(imageType, idx, 'done')

def knn(trainImages, trainLabels, testImages, testLabels, k = 59, dist = 'manhattan'):
    trainImages = np.array(trainImages)
    nr, height, width, dim = trainImages.shape
    trainImages = trainImages.reshape(nr, height * width * dim)

    testImages = np.array(testImages)
    nr, height, width, dim = testImages.shape
    testImages = testImages.reshape(nr, height * width * dim)

    scaler = StandardScaler()

    scaler.fit(trainImages)

    trainImages = scaler.transform(trainImages)
    testImages = scaler.transform(testImages)

    classifier = KNeighborsClassifier(n_neighbors = k, metric = dist)

    classifier.fit(trainImages, trainLabels)

    pred = classifier.predict(testImages)

    print(metrics.accuracy_score(testLabels, pred))

    print(metrics.classification_report(testLabels, pred))

    print(metrics.confusion_matrix(testLabels, pred))

    return metrics.accuracy_score(testLabels, pred)

def findOptimalK():
    accuracy = []
    for i in range(1, 100):
        accuracy.append(knn(trainImages, trainLabels, testImages, testLabels, i))
    accuracy = np.array(accuracy)

    plt.figure(figsize = (10, 6))
    plt.plot(range(1, 100), accuracy, color = 'red', linestyle = 'dashed', marker = 'o', markerfacecolor = 'blue', markersize = 8)
    plt.title('Manhattan Distance')
    plt.xlabel('K')
    plt.ylabel('Accuracy')

    plt.show()

def svm(trainImages, trainLabels, testImages, testLabels):
    trainImages = np.array(trainImages)
    nr, height, width, dim = trainImages.shape
    trainImages = trainImages.reshape(nr, height * width * dim)

    testImages = np.array(testImages)
    nr, height, width, dim = testImages.shape
    testImages = testImages.reshape(nr, height * width * dim)

    scaler = StandardScaler()

    scaler.fit(trainImages)

    trainImages = scaler.transform(trainImages)
    testImages = scaler.transform(testImages)

    classifier = SVC(kernel = 'linear')

    classifier.fit(trainImages, trainLabels)

    pred = classifier.predict(testImages)

    print(metrics.accuracy_score(testLabels, pred))

    print(metrics.classification_report(testLabels, pred))

    print(metrics.confusion_matrix(testLabels, pred))

def naiveBayes(trainImages, trainLabels, testImages, testLabels):
    trainImages = np.array(trainImages)
    nr, height, width, dim = trainImages.shape
    trainImages = trainImages.reshape(nr, height * width * dim)

    testImages = np.array(testImages)
    nr, height, width, dim = testImages.shape
    testImages = testImages.reshape(nr, height * width * dim)

    scaler = StandardScaler()

    scaler.fit(trainImages)

    trainImages = scaler.transform(trainImages)
    testImages = scaler.transform(testImages)

    classifier = GaussianNB()

    classifier.fit(trainImages, trainLabels)

    pred = classifier.predict(testImages)

    print(metrics.accuracy_score(testLabels, pred))

    print(metrics.classification_report(testLabels, pred))

    print(metrics.confusion_matrix(testLabels, pred))

def kMeansClustering(trainImages):
    trainImages = np.array(trainImages)
    nr, height, width, dim = trainImages.shape
    trainImages = trainImages.reshape(nr, height * width * dim)

    """
    plt.scatter(trainImages[:, 0], trainImages[:, 1])
    plt.show()
    """

    clustering = KMeans(n_clusters = 9)

    clustering.fit(trainImages)

    plt.scatter(trainImages[:, 0], trainImages[:, 1], c = clustering.labels_, cmap = 'rainbow')
    plt.show()

def selfOrganizingMaps(images, labels):
    images = np.array(images)
    nr, height, width, dim = images.shape
    images = images.reshape(nr, height * width * dim)

    scaler = StandardScaler()

    scaler.fit(images)

    images = scaler.transform(images)

    cls = SOM(m = 9, n = 1, dim = height * width * dim)

    cls.fit(images)

    pred = cls.predict(images)

    fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize = (5, 7))
    x = images[:, 0]
    y = images[:, 1]
    colors = ['black', 'blue', 'brown', 'green', 'pink', 'red', 'silver', 'white', 'yellow']

    ax[0].scatter(x, y, c = labels, cmap = ListedColormap(colors))
    ax[0].title.set_text('Actual Classes Standardization')
    ax[1].scatter(x, y, c = pred, cmap = ListedColormap(colors))
    ax[1].title.set_text('SOM Predictions Standardization')

    plt.show()

def vgg16(trainImages, trainLabels, testImages, testLabels):
    trainImages = np.array(trainImages)
    trainLabels = np.array(trainLabels)
    testImages = np.array(testImages)
    testLabels = np.array(testLabels)

    model = Sequential()

    model.add(Conv2D(input_shape = (224, 224, 3), filters = 64, kernel_size = (3, 3), padding = "same", activation = "relu"))
    model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding = "same", activation = "relu"))

    model.add(MaxPool2D(pool_size = (2, 2), strides = (2, 2)))

    model.add(Conv2D(filters = 128, kernel_size = (3, 3), padding = "same", activation = "relu"))
    model.add(Conv2D(filters = 128, kernel_size = (3, 3), padding = "same", activation = "relu"))

    model.add(MaxPool2D(pool_size = (2, 2), strides = (2, 2)))

    model.add(Conv2D(filters = 256, kernel_size = (3, 3), padding = "same", activation = "relu"))
    model.add(Conv2D(filters = 256, kernel_size = (3, 3), padding = "same", activation = "relu"))
    model.add(Conv2D(filters = 256, kernel_size = (3, 3), padding = "same", activation = "relu"))

    model.add(MaxPool2D(pool_size = (2, 2), strides = (2, 2)))

    model.add(Conv2D(filters = 512, kernel_size = (3, 3), padding = "same", activation = "relu"))
    model.add(Conv2D(filters = 512, kernel_size = (3, 3), padding = "same", activation = "relu"))
    model.add(Conv2D(filters = 512, kernel_size = (3, 3), padding = "same", activation = "relu"))

    model.add(MaxPool2D(pool_size = (2, 2), strides = (2, 2)))

    model.add(Conv2D(filters = 512, kernel_size = (3, 3), padding = "same", activation = "relu"))
    model.add(Conv2D(filters = 512, kernel_size = (3, 3), padding = "same", activation = "relu"))
    model.add(Conv2D(filters = 512, kernel_size = (3, 3), padding = "same", activation = "relu"))

    model.add(MaxPool2D(pool_size = (2, 2), strides = (2, 2)))

    model.add(Flatten())

    model.add(Dense(units = 4096, activation = "relu"))
    model.add(Dense(units = 4096, activation = "relu"))

    model.add(Dense(units = 9, activation = "softmax"))

    opt = Adam(learning_rate = 0.001)

    model.compile(optimizer = opt, loss = keras.losses.sparse_categorical_crossentropy, metrics = ['accuracy'])

    model.fit(trainImages, trainLabels, epochs = 10)

    pred = model.predict(testImages)

    print(pred)

def cnn(trainImages, trainLabels, testImages, testLabels):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    tensorflow.random.set_seed(1)

    trainImages = np.array(trainImages)
    trainLabels = np.array(trainLabels)
    testImages = np.array(testImages)
    testLabels = np.array(testLabels)

    trainImages = trainImages / 255
    testImages = testImages / 255

    model = Sequential()

    model.add(Conv2D(filters = 32, kernel_size = (3, 3), padding = 'same', activation = 'relu', input_shape = (224, 224, 3)))
    model.add(MaxPool2D(pool_size = (2, 2), strides = (2, 2)))

    model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
    model.add(MaxPool2D(pool_size = (2, 2), strides = (2, 2)))

    model.add(Conv2D(filters = 128, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
    model.add(MaxPool2D(pool_size = (2, 2), strides = (2, 2)))

    model.add(Flatten())

    model.add(Dense(units = 256, activation = 'relu'))

    model.add(Dense(units = 9, activation = 'softmax'))

    opt = Adam(learning_rate = 0.001)

    model.compile(optimizer = opt, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

    model.fit(trainImages, trainLabels, epochs = 15, batch_size = 64)

    predictionResult = model.predict(testImages)

    pred = []
    for i in range(len(predictionResult)):
        pred.append(np.argmax(predictionResult[i], axis = -1))

    vehicles = ['Black Vehicles', 'Blue Vehicles', 'Brown Vehicles', 'Green Vehicles', 'Pink Vehicles', 'Red Vehicles', 'Silver Vehicles', 'White Vehicles', 'Yellow Vehicles']

    print('Accuracy: ', metrics.accuracy_score(testLabels, pred))

    print(metrics.classification_report(testLabels, pred, target_names = vehicles))

    print(metrics.confusion_matrix(testLabels, pred))

    i = 0
    for image in sorted(glob.glob('C:/Users/razva/OneDrive/Desktop/Vehicle Color Recognition/dataset/newTestPCA/*.jpg')):
        print(image, ' : ', pred[i])
        i += 1

def appFlask():
    app = Flask(__name__)

    print('Model loaded. Check http://127.0.0.1:5000/')

    return app

def predictModel(imagePath, trainImages, trainLabels):
    trainImages = np.array(trainImages)
    trainLabels = np.array(trainLabels)

    trainImages = trainImages / 255

    model = Sequential()

    model.add(Conv2D(filters = 32, kernel_size = (3, 3), padding = 'same', activation = 'relu', input_shape = (224, 224, 3)))
    model.add(MaxPool2D(pool_size = (2, 2), strides = (2, 2)))

    model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
    model.add(MaxPool2D(pool_size = (2, 2), strides = (2, 2)))

    model.add(Conv2D(filters = 128, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
    model.add(MaxPool2D(pool_size = (2, 2), strides = (2, 2)))

    model.add(Flatten())

    model.add(Dense(units = 256, activation='relu'))

    model.add(Dense(units = 9, activation='softmax'))

    opt = Adam(learning_rate = 0.001)

    model.compile(optimizer = opt, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

    model.fit(trainImages, trainLabels, epochs = 15, batch_size = 64)

    detector = ObjectDetection()

    modelPath = 'C:/Users/razva/OneDrive/Desktop/Vehicle Color Recognition/yolo.h5'

    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(modelPath, )
    detector.loadModel(detection_speed = 'flash')

    returnedImage, detection = detector.detectObjectsFromImage(input_image = imagePath, output_type = 'array')

    image = Image.open(imagePath)

    sol = []
    for i in range(len(detection)):
        imageX = image.crop(detection[i]['box_points'])
        imageX = imageX.resize((224, 224))
        imageX = np.array(imageX)
        imageX = imageX.reshape(1, 224, 224, 3)
        imageX = imageX / 255

        pred = model.predict(imageX)

        sol.append((pred, detection[i]['name'], detection[i]['box_points']))

    return sol

"""
vehicleDetector('C:/Users/razva/OneDrive/Desktop/Vehicle Color Recognition/dataset/train/*.jpg',
                'C:/Users/razva/OneDrive/Desktop/Vehicle Color Recognition/dataset/newTrain/')
vehicleDetector('C:/Users/razva/OneDrive/Desktop/Vehicle Color Recognition/dataset/test/*.jpg',
                'C:/Users/razva/OneDrive/Desktop/Vehicle Color Recognition/dataset/newTest/')

principalComponentAnalysis('C:/Users/razva/OneDrive/Desktop/Vehicle Color Recognition/dataset/newTrain/*.jpg',
                            'C:/Users/razva/OneDrive/Desktop/Vehicle Color Recognition/dataset/newTrainPCA/')
principalComponentAnalysis('C:/Users/razva/OneDrive/Desktop/Vehicle Color Recognition/dataset/newTest/*.jpg',
                           'C:/Users/razva/OneDrive/Desktop/Vehicle Color Recognition/dataset/newTestPCA/')
"""

def inputDataset(inputPathTrainImages, inputPathTrainLabels, inputPathTestImages, inputPathTestLabels):
    trainImages = []
    for imagePath in sorted(glob.glob(inputPathTrainImages)):
        image = imageio.imread(imagePath, pilmode = 'RGB')
        trainImages.append(image)

    trainLabels = []
    f = open(inputPathTrainLabels, 'r')
    lines = f.readlines()
    for line in lines:
        trainLabels.append(int(line))

    testImages = []
    for imagePath in sorted(glob.glob(inputPathTestImages)):
        image = imageio.imread(imagePath, pilmode = 'RGB')
        testImages.append(image)

    testLabels = []
    f = open(inputPathTestLabels, 'r')
    lines = f.readlines()
    for line in lines:
        testLabels.append(int(line))

    return trainImages, trainLabels, testImages, testLabels

trainImages, trainLabels, testImages, testLabels = inputDataset('C:/Users/razva/OneDrive/Desktop/Vehicle Color Recognition/dataset/newTrainPCA/*.jpg',
                                                                'C:/Users/razva/OneDrive/Desktop/Vehicle Color Recognition/dataset/trainLabel.txt',
                                                                'C:/Users/razva/OneDrive/Desktop/Vehicle Color Recognition/dataset/NewTestPCA/*.jpg',
                                                                'C:/Users/razva/OneDrive/Desktop/Vehicle Color Recognition/dataset/testLabel.txt')


print('Choose: \n1). Naive Bayes \n2). KNN \n3). SVM \n4). CNN \n5). CNN with Flask \n')

opt = int(input('Your option: '))

if opt == 1:
    naiveBayes(trainImages, trainLabels, testImages, testLabels)
elif opt == 2:
    knn(trainImages, trainLabels, testImages, testLabels)
elif opt == 3:
    svm(trainImages, trainLabels, testImages, testLabels)
elif opt == 4:
    cnn(trainImages, trainLabels, testImages, testLabels)
elif opt == 5:
    app = appFlask()

    @app.route('/', methods=['GET'])
    def index():
        # Main page
        return render_template('index.html')

    @app.route('/predict', methods=['GET', 'POST'])
    def upload():
        if request.method == 'POST':
            # Get the file from post request
            f = request.files['file']

            # Save the file to ./uploads
            basepath = os.path.dirname(__file__)
            filePath = os.path.join(basepath, 'uploads', secure_filename(f.filename))
            f.save(filePath)

            sol = predictModel(filePath, trainImages, trainLabels)
            output = ''

            for i in range(len(sol)):
                preds, vehicleType, boxPoints = sol[i]
                result = np.argmax(preds, axis = -1)

                if result[0] == 0:
                    color = 'black'
                elif result[0] == 1:
                    color = 'blue'
                elif result[0] == 2:
                    color = 'brown'
                elif result[0] == 3:
                    color = 'green'
                elif result[0] == 4:
                    color = 'pink'
                elif result[0] == 5:
                    color = 'red'
                elif result[0] == 6:
                    color = 'silver'
                elif result[0] == 7:
                    color = 'white'
                elif result[0] == 8:
                    color = 'yellow'

                output += color + ';' + vehicleType + ';[' + str(boxPoints[0]) + ', ' + str(boxPoints[1]) + ', ' + str(
                    boxPoints[2]) + ', ' + str(boxPoints[3]) + ']|'

            output = output[:-1]
            return output

        return None

    app.run(debug = True)
