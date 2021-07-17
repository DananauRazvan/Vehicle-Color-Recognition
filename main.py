import glob
import imageio
import keras
import tensorflow as tf
from imageai.Detection import ObjectDetection
from PIL import Image

def vehicleDetector(input_path, output_path):
    detector = ObjectDetection()

    model_path = 'C:/Users/razva/OneDrive/Desktop/Vehicle Color Recognition/yolo.h5'

    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(model_path, )
    detector.loadModel(detection_speed = "flash")

    for imagePath in glob.glob(input_path):
        returned_image, detection = detector.detectObjectsFromImage(input_image = imagePath, output_type = 'array')
        max_percentage_probability = 0
        index = -1
        i = -1
        ok = False
        for eachItem in detection:
            i += 1
            if (eachItem['name'] == 'car' or eachItem['name'] == 'truck' or eachItem['name'] == 'motorcycle') and eachItem['percentage_probability'] > max_percentage_probability:
                index = i
                max_percentage_probability = eachItem['percentage_probability']
                ok = True

        if ok:
            print(detection[index]['name'], " : ", detection[index]['percentage_probability'], " : ",
                  detection[index]['box_points'])
            image = Image.open(imagePath)
            box = detection[index]['box_points']
            image = image.crop(box)
            image.save(output_path + imagePath[71:])
        else:
            print('undetectable')
            image = Image.open(imagePath)
            image.save(output_path + imagePath[71:])
            continue

def input(input_path_train_images, input_path_train_labels, input_path_test_images, input_path_test_labels):
    vehicleDetector('C:/Users/razva/OneDrive/Desktop/Vehicle Color Recognition/dataset/train/*.jpg',
                    'C:/Users/razva/OneDrive/Desktop/Vehicle Color Recognition/dataset/newTrain/')
    vehicleDetector('C:/Users/razva/OneDrive/Desktop/Vehicle Color Recognition/dataset/test/*.jpg',
                    'C:/Users/razva/OneDrive/Desktop/Vehicle Color Recognition/dataset/newTest/')

    trainImages = []
    for imagePath in glob.glob(input_path_train_images):
        image = imageio.imread(imagePath)
        trainImages.append(image)

    trainLabels = []
    f = open(input_path_train_labels, 'r')
    lines = f.readlines()
    for line in lines:
        trainLabels.append(int(line))

    testImages = []
    for imagePath in glob.glob(input_path_test_images):
        image = imageio.imread(imagePath)
        testImages.append(image)

    testLabels = []
    f = open(input_path_test_labels, 'r')
    lines = f.readlines()
    for line in lines:
        testLabels.append(int(line))

    return trainImages, trainLabels, testImages, testLabels


trainImages, trainLabels, testImages, testLabels = input('C:/Users/razva/OneDrive/Desktop/Vehicle Color Recognition/dataset/newTrain/*jpg',
                                                         'C:/Users/razva/OneDrive/Desktop/Vehicle Color Recognition/dataset/trainLabel.txt',
                                                         'C:/Users/razva/OneDrive/Desktop/Vehicle Color Recognition/dataset/newTest/*.jpg',
                                                         'C:/Users/razva/OneDrive/Desktop/Vehicle Color Recognition/dataset/testLabel.txt')
