import glob
import imageio
import keras
import tensorflow
from imageai.Detection import ObjectDetection
from PIL import Image

trainImages = []
for imagePath in glob.glob('C:/Users/razva/OneDrive/Desktop/Vehicle-Color-Recognition/dataset/train/*jpg'):
     image = imageio.imread(imagePath)
     trainImages.append(image)

trainLabels = []
f = open('C:/Users/razva/OneDrive/Desktop/Vehicle-Color-Recognition/dataset/trainLabel.txt', 'r')
lines = f.readlines()
for line in lines:
     trainLabels.append(int(line))

testImages = []
for imagePath in glob.glob('C:/Users/razva/OneDrive/Desktop/Vehicle-Color-Recognition/dataset/test/*.jpg'):
     image = imageio.imread(imagePath)
     testImages.append(image)

testLabels = []
f = open('C:/Users/razva/OneDrive/Desktop/Vehicle-Color-Recognition/dataset/testLabel.txt', 'r')
lines = f.readlines()
for line in lines:
     testLabels.append(int(line))

detector = ObjectDetection()

model_path = 'C:/Users/razva/OneDrive/Desktop/Vehicle-Color-Recognition/yolo/models/yolo.h5'
input_path = 'C:/Users/razva/OneDrive/Desktop/Vehicle-Color-Recognition/yolo/input/train-black (8).jpg'
output_path = 'C:/Users/razva/OneDrive/Desktop/Vehicle-Color-Recognition/dataset/newTrain/'

detector.setModelTypeAsYOLOv3()
detector.setModelPath(model_path)
detector.loadModel()

for imagePath in glob.glob('C:/Users/razva/OneDrive/Desktop/Vehicle-Color-Recognition/dataset/train/*.jpg'):
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
          print(detection[index]['name'] , " : ", detection[index]['percentage_probability'], " : ", detection[index]['box_points'])
     else:
          print('undetectable')
          image = Image.open(imagePath)
          image.save(output_path + imagePath[71:])

     image = Image.open(imagePath)
     box = detection[index]["box_points"]
     image = image.crop(box)
     image.save(output_path + imagePath[71:])
