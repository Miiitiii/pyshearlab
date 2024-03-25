import cv2
def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print("Toc: start time not set")

import numpy as np
from scipy import ndimage as img
from scipy import io as sio
import matplotlib.pyplot as plt



from pySLFilters import *
from pySLUtilities import *
from pyShearLab2D import *

tic()
print("--SLExampleImageDenoising")
print("loading image...")

scales = 2
sigma = 1
thresholdingFactor = 1

# load data
X = cv2.imread("img.jpg" , 0)
# X = cv2.resize(X , (224,224))
# X = img.imread("barbara.jpg")[::4, ::4]
X = X.astype(float)

toc()

tic()
print("generating shearlet system...")
## create shearlets
shearletSystem = SLgetShearletSystem2D(0,X.shape[0], X.shape[1], scales)

toc()
tic()
print("decomposition, thresholding and reconstruction...")

# decomposition
coeffs = SLsheardec2D(X, shearletSystem)

# # thresholding
oldCoeffs = coeffs.copy()
weights = np.ones(coeffs.shape)

for j in range(len(shearletSystem["RMS"])):
    weights[:,:,j] = shearletSystem["RMS"][j]*np.ones((X.shape[0], X.shape[1]))
    
coeffs = np.real(coeffs)
zero_indices = (np.abs(coeffs) / (thresholdingFactor * weights * 1)) < 0.1
coeffs[zero_indices] = 0

# # reconstruction
# Xrec = SLshearrec2D(coeffs, shearletSystem)
# toc()
# PSNR = SLcomputePSNR(X,Xrec)
# print("PSNR: " + str(PSNR))
# #sio.savemat("PyShearLab_DenoisingExample.mat", {"weights": weights, "XPyNoisy": Xnoisy,
# # "XPyDenoised": Xrec, "PyPSNR": PSNR, "coeffThrPy": coeffs, "oldCoeffs": oldCoeffs})
# plt.gray()
# plt.imshow(Xrec)
# plt.colorbar()
# plt.show()

def convert_shearlet(image1):
    new_image = np.zeros((X.shape))


    def custom_range_to_0_255(image, custom_min, custom_max):
        """
        Converts an image from a custom float range to the standard 0-255 range.
        Assumes the input image values are in the custom range [custom_min, custom_max].

        Args:
            image (numpy.ndarray): Input image (float values).
            custom_min (float): Minimum value of the custom range.
            custom_max (float): Maximum value of the custom range.

        Returns:
            numpy.ndarray: Image with pixel values in the 0-255 range (uint8 format).
        """
        # Normalize to [0, 1] range
        normalized_image = (image - custom_min) / (custom_max - custom_min)

        # Scale to [0, 255]
        scaled_image = (normalized_image * 255).astype(np.uint8)

        return scaled_image

    custom_min_value = np.min(image1)
    custom_max_value = np.max(image1)
    custom_mean_value = np.mean(image1)

    image1 = custom_range_to_0_255(image1, custom_min_value, custom_max_value)

    for row_index in range(X.shape[0]):
        for pixle in range(X.shape[1]):
            if X[row_index , pixle] == 0:
                new_image[row_index , pixle] = 0
            else:
                new_image[row_index , pixle] = image1[row_index , pixle]
                

    plt.gray()
    plt.imshow(new_image)
    plt.colorbar()
    plt.show()

print(coeffs.shape[2])
for i in range(coeffs.shape[2]):
    image1 = np.abs(coeffs[:,:,i])
    convert_shearlet(image1)