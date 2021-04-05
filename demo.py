import cv2
import torch
import torchvision.models as models
import matplotlib.cm as cm
from CAMERAS import CAMERAS
from torchvision import transforms
import numpy as np

normalizeTransform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
normalizeImageTransform = transforms.Compose([transforms.ToTensor(), normalizeTransform])

def loadImage(imagePath, imageSize):
    rawImage = cv2.imread(imagePath)
    rawImage = cv2.resize(rawImage, (224,) * 2, interpolation=cv2.INTER_LINEAR)
    rawImage = cv2.resize(rawImage, (imageSize,) * 2, interpolation=cv2.INTER_LINEAR)
    image = normalizeImageTransform(rawImage[..., ::-1].copy())
    return image, rawImage

def saveMapWithColorMap(filename, map, image):
    cmap = cm.jet_r(map)[..., :3] * 255.0
    map = (cmap.astype(np.float) + image.astype(np.float)) / 2
    cv2.imwrite(filename, np.uint8(map))

def computeAndSaveMaps():
    model = models.resnet18(pretrained=True)
    model.eval()
    model = model.cuda()

    cameras = CAMERAS(model, targetLayerName="layer4")
    file = "./cat_dog.png"

    image, rawImage = loadImage(file, imageSize=224)
    image = torch.unsqueeze(image, dim=0)

    saliencyMap = cameras.run(image, classOfInterest=243).cpu()
    saveMapWithColorMap("./Results/bullMastif.png", saliencyMap, rawImage)

    saliencyMap = cameras.run(image, classOfInterest=281).cpu()
    saveMapWithColorMap("./Results/tabbyCat.png", saliencyMap, rawImage)

if __name__ == '__main__':
    computeAndSaveMaps()

