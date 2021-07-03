from KerasMap.CAMERASKeras import CAMERASKeras
from tensorflow import keras
import numpy as np
import matplotlib.cm as cm
import cv2

def saveCAMERAS(originalImagePath, heatmap, saveAsFileName):
    img = keras.preprocessing.image.load_img(originalImagePath, target_size=(224, 224), interpolation="bilinear")
    img = keras.preprocessing.image.img_to_array(img)

    cmap = cm.jet_r(heatmap)[..., :3]*255
    superimposedImage = (cmap.astype(np.float) + img.astype(np.float)) / 2
    cv2.imwrite(saveAsFileName, np.uint8(superimposedImage))

ImagePath = './cat_dog.png'

cameras = CAMERASKeras(modelArchitecture="ResNet50", targetLayerName="conv5_block3_out")

saliency = cameras.run(imagePath=ImagePath, classOfInterest=243)
saveCAMERAS(ImagePath, saliency, "./Results/bullMastif.png")

saliency = cameras.run(imagePath=ImagePath, classOfInterest=281)
saveCAMERAS(ImagePath, saliency, "./Results/tabbyCat.png")