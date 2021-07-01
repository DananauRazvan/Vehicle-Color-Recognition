import glob
import imageio
import keras
import tensorflow
from imageai.Detection import ObjectDetection

trainImages = []
for imagePath in glob.glob('C:/Users/razva/OneDrive/Desktop/dataset/train/*.jpg'):
     image = imageio.imread(imagePath)
     trainImages.append(image)

trainLabels = []
f = open('C:/Users/razva/OneDrive/Desktop/dataset/trainLabel.txt', 'r')
lines = f.readlines()
for line in lines:
     trainLabels.append(int(line))

testImages = []
for imagePath in glob.glob('C:/Users/razva/OneDrive/Desktop/dataset/test/*.jpg'):
     image = imageio.imread(imagePath)
     testImages.append(image)

testLabels = []
f = open('C:/Users/razva/OneDrive/Desktop/dataset/testLabel.txt', 'r')
lines = f.readlines()
for line in lines:
     testLabels.append(int(line))

detector = ObjectDetection()

model_path = 'C:/Users/razva/OneDrive/Desktop/Facultate/Anul II/IA - Python/retina/models/yolo-tiny.h5'
input_path = 'C:/Users/razva/OneDrive/Desktop/Facultate/Anul II/IA - Python/retina/input/test47.jpg'
output_path = 'C:/Users/razva/OneDrive/Desktop/Facultate/Anul II/IA - Python/retina/output/newimage2.jpg'

detector.setModelTypeAsTinyYOLOv3()
detector.setModelPath(model_path)
detector.loadModel()
detection = detector.detectObjectsFromImage(input_image=input_path, output_image_path=output_path)

for eachItem in detection:
    print(eachItem["name"] , " : ", eachItem["percentage_probability"])