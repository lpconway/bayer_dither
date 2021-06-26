# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 20:17:52 2021

@author: Louis
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def _load_image(infilename):
    """
    Loads an image as a numpy array.

    Parameters
    ----------
    infilename : string
        The location of the source file.

    Returns
    -------
    The source image as a numpy array.

    """
    img = Image.open(infilename)
    img.load()
    data = np.asarray(img, dtype="int32")
    return(data)


def _save_image(dithered_image, outfilename):
    """
    Saves a numpy array as an image.

    Parameters
    ----------
    dithered_image : numpy array
        The numpy array to be saved.
    outfilename : string
        The path of the output file.

    Returns
    -------
    None.

    """
    depth = np.min([3, dithered_image.shape[2]])
    output = dithered_image[:, :, :depth]
    img = Image.fromarray(output.astype(np.uint8))
    img.save(outfilename)


def _plot_image(image):
    """
    Plots numpy array as an image, returns fig, axis.
    """
    fig, axis = plt.subplots(dpi=300)
    axis.imshow(image, cmap='gray')
    axis.axis('off')
    return(fig, axis)


def _bayer(n):
    """
    Recursively generates Bayer matrices.

    Parameters
    ----------
    n : int
        Bayer matrix level.

    Returns
    -------
    Bayer matrix of level n as a numpy array.

    """
    b_0 = np.array([[0, 2],
                    [3, 1]])
    if n == 0:
        return(b_0)
    order = 2**(n+1)
    # Create a matrix to hold the resulting Bayer matrix
    result = np.zeros((order, order))
    # Recursively generate the Bayer matrix of lower level
    sub_array = _bayer(n-1)
    # Create the Bayer matrix from the lower-level Bayer matrix
    result[0:order//2, 0:(order//2)] = 4 * sub_array + 0
    result[0:order//2, order//2:order] = 4 * sub_array + 2
    result[order//2:order, 0:order//2] = 4 * sub_array + 3
    result[order//2:order, order//2:order] = 4 * sub_array + 1
    return(result)


def _bayer_matrix(n):
    """
    Normalises the Bayer matrix.
    """
    return(_bayer(n) / (2 ** (2 * n + 2)))


def _create_bayer_threshold_map(height, width, n):
    """
    Tiles the Bayer matrix to create a threshold map of the appropriate dimensions.

    Parameters
    ----------
    height : int
        Height of the threshold map.
    width : int
        Width of the threshold map.
    n : int
        Bayer matrix level

    Returns
    -------
    Bayer matrix threshold map as a numpy array

    """
    
    # Generate the Bayer matrix of appropriate level
    matrix = _bayer_matrix(n)
    
    # Create matrix to hold the resulting threshold map
    threshold_map = np.zeros((height, width))
    
    # Calculate the order of the matrix
    order = 2 ** (n + 1)
    
    # Calculate the number of complete Bayer matrices that can be tiled on to
    # the threshold map, and the excess space that mut be tiled with partial matrices
    widths = width // order
    heights = height // order
    width_excess = width % order
    height_excess = height % order
    
    # Tile the matrices on to the threshold map
    for x in range(heights):
        for y in range(widths):
            threshold_map[x*order:(x+1)*order, y*order:(y+1)*order] = matrix
    
    # Fill in the residual space, if there is any
    if width_excess > 0:
        for x in range(heights):
            threshold_map[x*order:(x+1)*order, widths*order:widths*order+width_excess] = matrix[:,:width_excess]
    if height_excess > 0:
        for y in range(widths):
            threshold_map[heights * order:heights * order + height_excess, y * order:(y + 1) * order] = matrix[:height_excess, :]
    if (height_excess > 0) and (width_excess > 0):
        threshold_map[heights * order:heights * order + height_excess, widths * order:widths * order + width_excess] = matrix[:height_excess,:width_excess]
    return(threshold_map)


def dither_image(path, n=2, output_path=None):
    """
    
    Bayer dithers an image.
    
    Parameters
    ----------
    path : string
        Path to the source image.
    n : int, optional
        Bayer matrix level. The default is 2.
    output_path : string, optional
        If provided, the resulting image is saved to this path. The default is None.

    Returns
    -------
    fig, axis

    """
    # Load and normalise source image
    image = _load_image(path) / 255
    
    # Create the Bayer matrix threshold map
    threshold_map = _create_bayer_threshold_map(image.shape[0], image.shape[1], n)
    
    # Create a mtrix to hold the resulting dithered image
    dithered_image = np.zeros(image.shape)
    
    # Dither each layer in the image in turn
    for layer in range(image.shape[2]):
        dithered_image[:,:,layer] = image[:,:,layer] > threshold_map
    
    if output_path:
        _save_image(dithered_image * 255, output_path)

    return(_plot_image(dithered_image))