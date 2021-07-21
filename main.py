import glob
import imageio
import keras
import tensorflow
from imageai.Detection import ObjectDetection
from PIL import Image
from sklearn.decomposition import PCA
import cv2

def vehicleDetector(inputPath, outputPath):
    detector = ObjectDetection()

    modelPath = 'C:/Users/razva/OneDrive/Desktop/Vehicle Color Recognition/yolo.h5'

    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(modelPath, )
    detector.loadModel(detection_speed = 'flash')

    for imagePath in glob.glob(inputPath):
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

        if ok:
            print(detection[index]['name'], " : ", detection[index]['percentage_probability'], " : ", detection[index]['box_points'])
            image = Image.open(imagePath)
            box = detection[index]['box_points']
            image = image.crop(box)
            image.save(outputPath + imagePath[71:])
        else:
            print('undetectable')
            image = Image.open(imagePath)
            image.save(outputPath + imagePath[71:])
            continue

def principalComponentAnalysis(inputPath, outputPath):
    for imagePath in glob.glob(inputPath):
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
        else:
            name = imagePath[74:]

        cv2.imwrite(outputPath + name, imageReduced * 255)

def input(inputPathTrainImages, inputPathTrainLabels, inputPathTestImages, inputPathTestLabels):
    trainImages = []
    for imagePath in glob.glob(inputPathTrainImages):
        image = imageio.imread(imagePath)
        trainImages.append(image)

    trainLabels = []
    f = open(inputPathTrainLabels, 'r')
    lines = f.readlines()
    for line in lines:
        trainLabels.append(int(line))

    testImages = []
    for imagePath in glob.glob(inputPathTestImages):
        image = imageio.imread(imagePath)
        testImages.append(image)

    testLabels = []
    f = open(inputPathTestLabels, 'r')
    lines = f.readlines()
    for line in lines:
        testLabels.append(int(line))

    return trainImages, trainLabels, testImages, testLabels

vehicleDetector('C:/Users/razva/OneDrive/Desktop/Vehicle Color Recognition/dataset/train/*.jpg',
                'C:/Users/razva/OneDrive/Desktop/Vehicle Color Recognition/dataset/newTrain/')
vehicleDetector('C:/Users/razva/OneDrive/Desktop/Vehicle Color Recognition/dataset/test/*.jpg',
                'C:/Users/razva/OneDrive/Desktop/Vehicle Color Recognition/dataset/newTest/')

principalComponentAnalysis('C:/Users/razva/OneDrive/Desktop/Vehicle Color Recognition/dataset/newTrain/*.jpg',
                           'C:/Users/razva/OneDrive/Desktop/Vehicle Color Recognition/dataset/newTrainPCA/')
principalComponentAnalysis('C:/Users/razva/OneDrive/Desktop/Vehicle Color Recognition/dataset/newTest/*.jpg',
                           'C:/Users/razva/OneDrive/Desktop/Vehicle Color Recognition/dataset/newTestPCA/')