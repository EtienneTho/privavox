# Written by Etienne Thoret (2020)

import numpy as np
import random
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from scipy import stats

N = 28
# array = np.random.rand(N,N)
# array = np.zeros((N,N))
# array[1,1] = 1
# array[20,20] = 1
# fft_ = np.fft.fft2(array)
# print(fft_.shape)
# ifft_ = np.fft.ifft2(array)
# plt.imshow(np.abs(ifft_))
# plt.show()


def gabor(N=28,Nit=100,Nnoise=1):
	arrayTot = np.zeros((Nnoise,N,N))
	oneArray = np.zeros((N,N))
	oneArray[int(N/2),int(N/2)] = 1
	oneArray = gaussian_filter(oneArray,sigma=(10,10))
	oneArray -= np.min(oneArray)
	oneArray /= np.max(oneArray)
	for iNoise in range(Nnoise):
		array = np.zeros((N,N))
		for iIt in range(Nit):
			fft_ = np.zeros((N,N))
			fft_[1,1] = 1
			idx = np.random.randint(8, size=2)
			fft_[idx[0]][idx[1]] = np.random.randn()
			fft_ = gaussian_filter(fft_,sigma=(.1,.1))
			fft_ =  np.rot90(np.fliplr(fft_),k=np.random.randint(4))
			ifft_ = np.abs(np.fft.ifft2(np.multiply(fft_,np.exp(2*np.pi*np.random.rand(N,N)))))
			array += ifft_ 
			# plt.imshow(ifft_)
			# plt.show()

		# plt.imshow(stats.zscore(array) * oneArray)
		# plt.show()
		array -= np.min(np.abs(array))
		array /= np.max(array)
		array = array * 2 - 1		
		arrayTot[iNoise][:][:] = array * oneArray
	return arrayTot

def fractal_noise(N=28,Nit=100,Nnoise=1):
	arrayTot = np.zeros((Nnoise,N,N))
	oneArray = np.zeros((N,N))
	oneArray[int(N/2),int(N/2)] = 1
	oneArray = gaussian_filter(oneArray,sigma=(10,10))
	oneArray -= np.min(oneArray)
	oneArray /= np.max(oneArray)
	for iNoise in range(Nnoise):
		array = np.zeros((N,N))
		for iIt in range(Nit):
			fft_ = np.zeros((N,N))
			fft_[1,1] = 1
			idx = np.random.randint(8, size=2)
			fft_[idx[0]][idx[1]] = np.random.randn()
			fft_ = gaussian_filter(fft_,sigma=(.1,.1))
			fft_ =  np.rot90(np.fliplr(fft_),k=np.random.randint(4))
			ifft_ = np.abs(np.fft.ifft2(np.multiply(fft_,np.exp(2*np.pi*np.random.rand(N,N)))))
			array += ifft_ 
			# plt.imshow(ifft_)
			# plt.show()

		# plt.imshow(stats.zscore(array) * oneArray)
		# plt.show()
		array -= np.min(np.abs(array))
		array /= np.max(array)
		array = array * 2 - 1		
		arrayTot[iNoise][:][:] = array * oneArray
	return arrayTot


plt.imshow(gabor(N,1,2)[1][:][:])
plt.show()

