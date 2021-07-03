import numpy as np
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.layers import Input

class CAMERASKeras():
    def __init__(self, modelArchitecture, targetLayerName, inputResolutions=None):
        self.modelArch = modelArchitecture
        self.inputResolutions = inputResolutions

        if self.inputResolutions is None:
            self.inputResolutions = list(range(224, 1000, 100))

        self.featureDict = {}
        self.gradientsDict = {}
        self.targetLayerName = targetLayerName

    def _readImage(self, img_path, size):
        img = keras.preprocessing.image.load_img(img_path, target_size=size, interpolation="bilinear")
        array = keras.preprocessing.image.img_to_array(img)
        array = np.expand_dims(array, axis=0)
        return array

    def _getGradientsAndActivations(self, imageNumpy, model, layerName, classOfInterest):
        grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layerName).output, model.output])

        with tf.GradientTape() as tape:
            layerActivation, preds = grad_model(imageNumpy)
            classChannel = preds[:, classOfInterest]

        grads = tape.gradient(classChannel, layerActivation)
        activations = layerActivation[0]
        return activations, grads

    def _recordActivationsAndGradients(self, inputResolution, imagePath, classOfInterest=None):
        preprocessInput =  keras.applications.__dict__[self.modelArch.lower()].preprocess_input
        imageArray = preprocessInput(self._readImage(imagePath, size=(inputResolution, inputResolution)))

        inputTensor = Input(shape=(inputResolution, inputResolution, 3))
        model = tf.keras.applications.__dict__[self.modelArch](input_tensor=inputTensor, weights='imagenet', include_top=True)
        model.layers[-1].activation = None

        activations, grads = self._getGradientsAndActivations(imageArray, model, self.targetLayerName, classOfInterest=classOfInterest)
        self.featureDict[inputResolution] = activations
        self.gradientsDict[inputResolution] = grads

    def _estimateSaliencyMap(self, classOfInterest):
        resizeLayer = tf.keras.layers.experimental.preprocessing.Resizing(height=self.inputResolutions[0], width=self.inputResolutions[0], interpolation="bilinear")

        upSampledFeatures = None
        upSampledGradients = None
        count = 0

        for resolution in self.featureDict.keys():
            activations = self.featureDict[resolution]
            grads = self.gradientsDict[resolution]

            if upSampledFeatures is None or upSampledGradients is None:
                upSampledFeatures = resizeLayer(activations).numpy()
                upSampledGradients = resizeLayer(grads).numpy()
            else:
                upSampledFeatures += resizeLayer(activations).numpy()
                upSampledGradients += resizeLayer(grads).numpy()

            count += 1

        fmaps = upSampledFeatures / count
        grads = upSampledGradients / count

        fmaps = fmaps.squeeze()
        grads = grads.squeeze()

        saliency = tf.keras.activations.relu((fmaps * grads).sum(axis=2)).numpy()
        saliency = saliency - saliency.min()
        saliency = saliency / saliency.max()

        return saliency

    def run(self, imagePath, classOfInterest=None):
        for index, inputResolution in enumerate(self.inputResolutions):
            self._recordActivationsAndGradients(inputResolution, imagePath, classOfInterest=classOfInterest)

        saliencyMap = self._estimateSaliencyMap(classOfInterest=classOfInterest)
        return saliencyMap
