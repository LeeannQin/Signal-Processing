# -*- coding: utf-8 -*-
"""
@author: Bo Wang
@file: DataAnalysisScript1.py
@time: 2021/12/29 14:39
"""
import os

import numpy as np
from matplotlib import pyplot as plt

from PyEMD import EMD
from data_analysis_python.FileDataUtil import readOriginalData
from data_analysis_python.FrequencyAnalysisUtil import toNumpyArray, convertToFrequency

def emdDenoise(xFreq):
    """
        Empirical mode decomposition (EMD) is employed to remove the noise and extract the trend of the signal.

        js: The cut-off mode, those IMFs less than the js th order are discarded.

        Essentially, the denoising capability of the EMD is achieved by reconstruction from the last M IMFs and the residue.
    """
    # Normalization process
    xFreq = normalization(xFreq)

    # Make EMD decomposition
    emd = EMD()
    imfs = emd.emd(xFreq)

    # Find the cut-off mode of IMFs
    js = cutoffMode_CMSE(imfs)

    # Reconstruction process
    processedXFreq = sum(imf for imf in imfs[js:, :])

    return processedXFreq

def consecutiveMSE(imf):
    """
    Calculate the consecutive mean square error (CMSE).
    This quantity measures the squared Euclidean distance between two consecutive reconstructions of the signal.
    The CMSE is reduced to the energy of the k th IMF.
    """
    L = len(imf)
    CMSE = sum([i ** 2 for i in imf]) / L
    return CMSE

def cutoffMode_CMSE(imfs):
    """
        N: The number of intrinsic mode functions(IMFs).
    """
    N = imfs.shape[0] - 1
    CMSEs = np.zeros(N - 1)

    for ji in range(N-1):
        CMSEs[ji] = consecutiveMSE(imfs[ji, :])

    CMSEs = CMSEs.tolist()
    js = CMSEs.index(min(CMSEs))

    return js

def normalization(x):
    """
    Normalization process.
    """
    return (x - np.mean(x)) / np.std(x)

if __name__ == '__main__':
    dataPath = "data"
    dataFiles = os.listdir(dataPath)

    for dataFile in dataFiles:
        dataName, dataExt = os.path.splitext(dataFile)
        if dataExt == ".xls":
            t, x = toNumpyArray(readOriginalData(dataFile, dataPath))
            xAve = np.mean(x, axis=0)
            freq, XFreq = toNumpyArray(convertToFrequency(t.tolist(), xAve.tolist(), 0.1, 10, isInDb=False))
            # Process data here.
            processedXFreq = emdDenoise(XFreq)
            # Plot.
            plt.figure(dataFile)
            plt.subplot(211)
            plt.plot(freq, XFreq)
            plt.title("Original spectrum")
            plt.subplot(212)
            plt.plot(freq, processedXFreq)
            plt.title("EMD denoised spectrum")
            plt.tight_layout()

    plt.show()