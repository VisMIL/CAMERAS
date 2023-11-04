# CAMERAS

The repository provides implementation of CAMERAS with PyTorch

*__CAMERAS: Enhanced Resolution And Sanity preserving Class Activation Mapping for image Saliency__. Jalwana, Akhtar, Mian, Bennamoun. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021.*

Bibtex:
@inproceedings{jalwana2021cameras,
  title={CAMERAS: Enhanced Resolution And Sanity preserving Class Activation Mapping for image saliency},
  author={Jalwana, Mohammad AAK and Akhtar, Naveed and Bennamoun, Mohammed and Mian, Ajmal},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={16327--16336},
  year={2021}
}


A brief over view of scheme is as below 

![Teaser.png](https://github.com/VisMIL/CAMERAS/blob/main/figs/Teaser.png)

## Requirements
We have tested the code over Ubuntu 18.04 and with the following versions of packages

* python 3.6
* torch 1.7.1
* torchvision 0.9.1
* numpy 1.19.4
* opencv 3.2.0
* matplotlib 3.3.4

All of our experiments utilized Titan V GPU. 

## Comparative Results
Architecture  | Attribution maps by different available techniques. 
------------- | -------------
ResNet     | ![resnet_l.png](https://github.com/VisMIL/CAMERAS/blob/main/figs/resnet_l.png)
DenseNet   | ![densenet_l.png](https://github.com/VisMIL/CAMERAS/blob/main/figs/densenet_l.png)
Inception  | ![inception_l.png](https://github.com/VisMIL/CAMERAS/blob/main/figs/inception_l.png)

## How to compute CAMERAS maps?
In order to compute CAMERAS saliency map, we need to create an object of _CAMERAS_ defined in  **CAMERAS.py**.  The constructor accepts three hyper-parameters. 
* Particular model (Pytorch) over which the saliency map is to be evaluated
* The choice of the layer to for computation of gradients and activations 
* The list of different resolutions of input

Lets first create and load the weights of a ImageNet pretrained model from torchvision package. 

```python
    # importing the models available in torchvision package
    import torchvision.models as models

    # Loading a pretrained ResNet-18 model in torchvision
    model = models.resnet18(pretrained=True)
    model.eval()
    model = model.cuda()
```
Once we have the model, we can select any of its layer and different resolutions to create CAMERAS object as below 

```python
    # import CAMERAS class
    from CAMERAS import CAMERAS
    
    # Creating CAMERAS object
    cameras = CAMERAS(model=model, targetLayerName="layer4", inputResolutions=[224, 324, 424, 524])

```
With the CAMERAS object and any image tensor (1xCxHxW), we can compute the saliency maps as below

```python
    # compute saliency map for a image tensor and any label of interest
    saliencyMap = cameras.run(image, classOfInterest=243)
```
The above steps can be used to compute the saliency maps for any image and label of choice. At the moment, the current implementation can only compute the saliency map for a single image at time and does not offer batch support. 

A complete example with image loading and saliency computation is given below. 

## Demo 

```python
import cv2
import torch
import torchvision.models as models
from CAMERAS import CAMERAS
from torchvision import transforms
import matplotlib.pyplot as plt

normalizeTransform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
normalizeImageTransform = transforms.Compose([transforms.ToTensor(), normalizeTransform])

def loadImage(imagePath, imageSize):
    rawImage = cv2.imread(imagePath)compute
    rawImage = cv2.resize(rawImage, (224,) * 2, interpolation=cv2.INTER_LINEAR)
    rawImage = cv2.resize(rawImage, (imageSize,) * 2, interpolation=cv2.INTER_LINEAR)
    image = normalizeImageTransform(rawImage[..., ::-1].copy())
    return image, rawImage

if __name__ == '__main__':
    model = models.resnet18(pretrained=True)
    model.eval()
    model = model.cuda()

    cameras = CAMERAS(model, targetLayerName="layer4")
    file = "./cat_dog.png"

    image, rawImage = loadImage(file, imageSize=224)
    image = torch.unsqueeze(image, dim=0)

    saliencyMap = cameras.run(image, classOfInterest=243).cpu()
    plt.imshow(saliencyMap)
    plt.show()



``` 
A comprehensive example is available in `demo.py` script. 

## Announcements !!!

We have released the first version to compute CAMERAS maps in Keras. 
A complete demo code can be found in `demoKeras.py`.

## Requirements
The code was tested over Ubuntu 18.04 with the following versions of packages

* tensorflow 2.4.1
* python 3.8.1
* matplotlib 3.1.3
* opencv 4.5.1
* numpy 1.18.1

## How to compute CAMERAS maps?

In order to compute CAMERAS saliency map, we need to create an object of _CAMERASKeras_ defined in  **KerasMap/CAMERASKeras.py**. This can be done as below 

```python
    # import CAMERASKeras class
    from KerasMap.CAMERASKeras import CAMERASKeras
    
    # Creating CAMERASKeras object
    cameras = CAMERASKeras(modelArchitecture="ResNet50", targetLayerName="conv5_block3_out")

```

Saliency is then computed by calling `run()` that requires the path of image file and label of the class of interest.

```python
    saliency = cameras.run(imagePath=ImagePath, classOfInterest=243)

```

A complete example is given below. 

## Demo 

```python
import matplotlib.pyplot as plt
from KerasMap.CAMERASKeras import CAMERASKeras

ImagePath = './cat_dog.png'
cameras = CAMERASKeras(modelArchitecture="ResNet50", targetLayerName="conv5_block3_out")
saliency = cameras.run(imagePath=ImagePath, classOfInterest=243)
plt.imshow(saliency)
plt.show()
``` 

Please check **demoKeras.py** for detailed example. 