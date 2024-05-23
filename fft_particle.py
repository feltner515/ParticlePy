import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
import glob
from PIL import Image



def threshold(img, threshold):
    img = img[:,:,0] #only necessary if image has rgb channels
    for n in range(0,img.shape[0]):
        for m in range(0,img.shape[1]):
            if img[n,m] > threshold:
                img[n,m] = 0
            else:
                img[n,m] = 1

    return img

def centroid(img):
    # Find the non-zero indices
    nonzero_indices = np.nonzero(img)

    # Calculate the centroid coordinates
    centroid_x = np.mean(nonzero_indices[1])
    centroid_y = np.mean(nonzero_indices[0])
    return centroid_x, centroid_y

def edge_detection_distances(img, centroid_x, centroid_y):
    # Calculate the gradient of the image
    gradient_x, gradient_y = np.gradient(img)

    # Find the edge pixels
    edge_pixels = np.sqrt(gradient_x**2 + gradient_y**2) > 0

    distance = np.zeros((img.shape[0], img.shape[1]))
    for n in range (0,img.shape[0]):
        for m in range (0,img.shape[1]):
            if edge_pixels[n,m] == True:
                distance[n,m] = np.sqrt((m - centroid_x)**2 + (n - centroid_y)**2)


    return distance


def convert_to_polar(img,distance, centroid_x, centroid_y):
    angles = np.array([])
    distances = np.array([])
    for n in range (0,img.shape[0]):
        for m in range (0,img.shape[1]):
            angle = np.arctan2(n - centroid_y, m - centroid_x)
            if distance[n,m] > 0:
                angles = np.append(angles, angle)
                distances = np.append(distances, distance[n,m])

    return angles, distances

def interpolate_polar(angles, distances):
    sorted_indices = np.argsort(angles)
    sorted_angles = angles[sorted_indices]
    sorted_distances = distances[sorted_indices]
    cs = interp1d(sorted_angles, sorted_distances)
    xs = np.linspace(sorted_angles[0], sorted_angles[-1], 1000)
    return xs, cs(xs)


                



def decompose_fft(data: list, threshold: float = 0.0):
    fft3 = np.fft.fft(data)
    x = np.arange(0, 2*np.pi, 2*np.pi / len(data))
    freqs = np.fft.fftfreq(len(x), 2*np.pi / len(x))
    recomb = np.zeros((len(x),))
    for i in range(len(fft3)):
        if abs(fft3[i]) / len(x) > threshold:
            sinewave = (
                1 
                / len(x) 
                * (
                    fft3[i].real 
                    * np.cos(freqs[i] * 2 * np.pi * x) 
                    - fft3[i].imag 
                    * np.sin(freqs[i] * 2 * np.pi * x)))
            recomb += sinewave
            plt.polar(x,recomb)
            plt.polar(x, sinewave)  # Plotting on a polar graph
    plt.show()

    plt.polar(x, recomb, label='Recombined')  # Plotting on a polar graph
    plt.polar(x, data, label='Original')  # Plotting on a polar graph
    plt.legend()  # Adding legend
    plt.show()
    return x, recomb
    

def master_particle_fft(img,threshold_fft, threshold_edge):
    img = threshold(img, threshold_edge)
    centroid_x, centroid_y = centroid(img)
    distance = edge_detection_distances(img, centroid_x, centroid_y)
    angles, distances_polar = convert_to_polar(img, distance, centroid_x, centroid_y)
    xs, cs = interpolate_polar(angles, distances_polar)

    return decompose_fft(cs, threshold_fft)

img = img = Image.open('B3P0X1.png')
img = np.array(img)
master_particle_fft(img, 2, 173)
