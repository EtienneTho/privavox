# Basic functions for probing classifiers with revcor and bubbles
#
# Written by Etienne Thoret (2020)

import numpy as np
import random
from scipy.ndimage import gaussian_filter
from scipy.stats import pointbiserialr
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pingouin as pg
from pingouin.effsize import compute_bootci
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, LinearRegression
from sklearn.metrics import r2_score
from sklearn.svm import SVC  
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy import stats
from perlin_numpy import generate_perlin_noise_3d, generate_fractal_noise_3d

# compute the canonical map for a given category (category) from the probed samples (tab) and their classification item (y_pred)
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
       idx = np.random.randint(6, size=2)
       fft_[idx[0]][idx[1]] = np.random.randn()
       fft_ = gaussian_filter(fft_,sigma=(.1,.1))
       fft_ =  np.rot90(np.fliplr(fft_),k=np.random.randint(4))
       ifft_ = np.abs(np.fft.ifft2(np.multiply(fft_,np.exp(1j*2*np.pi*np.random.rand(N,N)))))
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

def ComputeOneCanonicalMap(tab,y_pred,category):
   canonicalMap = (np.mean(np.asarray(tab[y_pred==category,:]),axis=0)-np.mean(np.asarray(tab[y_pred!=category,:]),axis=0))/np.sqrt(1/2*(np.std(np.asarray(tab[y_pred==category,:]),axis=0)**2+np.std(np.asarray(tab[y_pred!=category,:]),axis=0)**2))
   return canonicalMap

# compute the canonical maps for the different categories
def ComputeCanonicalMaps(tab,y_pred,dimOfinput):
   tabMaps = []
   for iCateg in range(np.unique(y_pred).shape[0]):
     category = np.unique(y_pred)[iCateg]
     tabMaps.append(ComputeOneCanonicalMap(tab,y_pred,category))
   return tabMaps

# compute the canonical maps for the different categories
def ComputeCanonicalMapsWithGroundTruth(tab,y_pred,y_true,dimOfinput):
   tabMaps = []
   print('ytrue'+str(np.unique(y_pred)))
   for iCateg in range(np.unique(y_true).shape[0]):
     category = np.unique(y_true)[iCateg]
     responses_ = np.zeros(tab.shape[0])
     
     tabTemp = tab[y_true==category,:]
     y_true_temp = y_true[y_true==category]
     y_pred_temp = y_pred[y_true==category]
     
     print(np.sum(y_pred_temp==y_true_temp)/y_pred_temp.shape[0])
     trueResponses = np.zeros(tabTemp.shape[0])
     categSamples = np.zeros(tabTemp.shape[0])

     trueResponses[y_pred_temp==y_true_temp] = 1

     tabMaps.append(ComputeOneCanonicalMapCorrelation(tabTemp,trueResponses,1))
   return tabMaps, np.unique(y_true)

# compute the canonical map for a given category (category) from the probed samples (tab) and their classification item (y_pred)
def ComputeOneCanonicalMapCorrelation(tab,y_pred,category):
   canonicalMap = np.zeros(tab.shape[1])
   ci95 = []
   zscore = np.zeros(tab.shape[1])
   pval = np.zeros(tab.shape[1])
   responses_ = np.zeros(tab.shape[0])
   responses_[y_pred==category] = 1

   for i in range(tab.shape[1]):
     corr_ = pg.corr(tab[:,i], responses_, method='pearson')
     canonicalMap[i] = corr_['r']
     # # print(np.asarray(corr_['CI95%'])[0][0])
     # bootci, dist_ = compute_bootci(tab[:,i], y=responses_, func='pearson', n_boot=500, return_dist=True, confidence=.95) 
     # # zscore[i] = corr_['r']/np.abs((bootci[0]-bootci[1]))*1.96
     # # ci95.append([np.abs(corr_['CI95%'][0][0]-corr_['r']),np.abs(corr_['CI95%'][0][1]-corr_['r'])])
     # ci95.append([np.abs(bootci[0]-corr_['r']),np.abs(bootci[1]-corr_['r'])])
     # pval[i] = corr_['p-val']

   # canonicalMap = (np.mean(np.asarray(tab[y_pred==category,:]),axis=0)-np.mean(np.asarray(tab[y_pred!=category,:]),axis=0))/np.sqrt(1/2*(np.std(np.asarray(tab[y_pred==category,:]),axis=0)**2+np.std(np.asarray(tab[y_pred!=category,:]),axis=0)**2))

   return canonicalMap#, np.asarray(ci95), pval, zscore

# compute the canonical maps for the different categories
def ComputeCanonicalMapsCorrelation(tab,y_pred,dimOfinput):
   tabMaps = []
   tabCI95 = []
   tabPval = []
   tabZscore = []
   for iCateg in range(np.unique(y_pred).shape[0]):
     category = np.unique(y_pred)[iCateg]
     maps = ComputeOneCanonicalMapCorrelation(tab,y_pred,category) # , ci95, pval, zscores
     tabMaps.append(maps)
     # tabCI95.append(ci95)
     # tabPval.append(pval)
     # tabZscore.append(zscore)
   return tabMaps#, tabCI95, tabPval, tabZscore

# compute the canonical map for a given category (category) from the probed samples (tab) and their classification item (y_pred)
def ComputeOneCanonicalMapCorrelationContinuous(tab,y_pred,y_proba,category):
   canonicalMap = np.zeros(tab.shape[1])
   responses_ = np.zeros(tab.shape[0])
   responses_[y_pred==category] = 1

   for i in range(tab.shape[1]):
     canonicalMap[i] = pg.corr(tab[:,i], y_proba[:,category], method='pearson')['r']

   # canonicalMap = (np.mean(np.asarray(tab[y_pred==category,:]),axis=0)-np.mean(np.asarray(tab[y_pred!=category,:]),axis=0))/np.sqrt(1/2*(np.std(np.asarray(tab[y_pred==category,:]),axis=0)**2+np.std(np.asarray(tab[y_pred!=category,:]),axis=0)**2))
   return canonicalMap

# compute the canonical maps for the different categories
def ComputeCanonicalMapsCorrelationContinuous(tab,y_pred,y_proba,dimOfinput):
   tabMaps = []
   for iCateg in range(np.unique(y_pred).shape[0]):
     category = np.unique(y_pred)[iCateg]
     tabMaps.append(ComputeOneCanonicalMapCorrelationContinuous(tab,y_pred,y_proba,category))
   return tabMaps

# compute the discriminative map for a given category (category) from the probed samples (tab) and their classification item (y_pred)
def ComputeOneDiscriminativeMap(tab,y_pred,category):
   dprimeTab = np.sum(np.asarray(tab[y_pred==category,:]),axis=0) / (np.sum(np.asarray(tab),axis=0))
   return dprimeTab

# compute the discriminative map for a given category (category) from the probed samples (tab) and their classification item (y_pred)
def ComputeOneDiscriminativeMapCorrelation(tab,y_pred,category):
   # dprimeTab = np.sum(np.asarray(tab[y_pred==category,:]),axis=0) / (np.sum(np.asarray(tab),axis=0))
   dprimeTab = np.zeros(tab.shape[1])
   responses_ = np.zeros(tab.shape[0])
   responses_[y_pred==category] = 1
   # clf = LinearRegression()
   clf = LogisticRegressionCV(class_weight='balanced',n_jobs=3)
   # clf = SVC(kernel='rbf', class_weight='balanced')

   # SVC(kernel='rbf').fit(tabStrf_pca, tabLabels)
   X_temp = tab[:,0].flatten().reshape(-1, 1)
   for i in range(tab.shape[1]):     


     # X = np.asarray(tab[:,i].flatten().reshape(-1, 1))
     # y = np.asarray(responses_.flatten().reshape(-1, 1).ravel())
     dprimeTab[i] = pg.corr(tab[:,i], responses_,method='pearson')['r']
     # X_temp = X     
     # clf.fit(X, y)
     # y_pred = clf.predict(X)
     # print(r2_score(responses_, clf.predict(tab[:,i].flatten().reshape(-1, 1))))
     # print(r2_score(y, y_pred))
     # dprimeTab[i] = r2_score(y, y_pred) ########
     # plt.scatter(X,y,s=.3)
     # plt.show()
     # print(dprimeTab[i])

     # print(pointbiserialr(responses_, tab[:,i]))
     # dprimeTab[i] = pointbiserialr(responses_, tab[:,i])[0]
     # if category == 7:
     #   print(dprimeTab[i])
     #   plt.scatter(responses_, tab[:,i])
     #   plt.show()
     
   return dprimeTab

# compute the discriminative map for a given category (category) from the probed samples (tab) and their classification item (y_pred)
def ComputeOneDiscriminativeMapCorrelationContinuous(tab,y_pred,y_proba,category):
   # dprimeTab = np.sum(np.asarray(tab[y_pred==category,:]),axis=0) / (np.sum(np.asarray(tab),axis=0))
   dprimeTab = np.zeros(tab.shape[1])

   for i in range(tab.shape[1]):
     dprimeTab[i] = pg.corr(tab[:,i], y_proba[:,category], method='pearson')['r']
     # if category == 1:
     #   print(dprimeTab[i])
     #   plt.scatter(tab[:,i],y_proba[:,category])
     #   plt.show()
   return dprimeTab

# compute the discriminative maps for the different categories
def ComputeDiscriminativeMaps(tab,y_pred,dimOfinput):
   tabMaps = []
   for iCateg in range(np.unique(y_pred).shape[0]):
     print(str(iCateg)+' over '+str(len(np.unique(y_pred)))+' classes')
     category = np.unique(y_pred)[iCateg]
     tabMaps.append(ComputeOneDiscriminativeMapCorrelation(tab,y_pred,category))
   return tabMaps

# compute the discriminative maps for the different categories
def ComputeGlobalDiscriminativeMaps(tab,y_pred,y_true,dimOfinput):
   tabMaps = ComputeOneDiscriminativeMapCorrelation(tab,y_pred==y_true,1)
   return tabMaps


# compute the discriminative maps for the different categories
def ComputeDiscriminativeMapsContinuous(tab,y_pred,y_proba,dimOfinput):
   tabMaps = []
   for iCateg in range(np.unique(y_pred).shape[0]):
     category = np.unique(y_pred)[iCateg]
     tabMaps.append(ComputeOneDiscriminativeMapCorrelationContinuous(tab,y_pred,y_proba,category))
   return tabMaps

# generate the bubble masks
def generateBubbleMask(dimOfinput=(28,28), nbMasks=10, probaBubbles=.1, bubbleSize=[4, 4]):
   nFeatures = np.product(dimOfinput) # np.asarray(dimOfinput)[0]*np.asarray(dimOfinput)[1]
   masks = np.random.rand(nbMasks,nFeatures)
   binaryMasks = np.random.rand(nbMasks,nFeatures)
   # nbBubbles = 10
   for iSample in range(nbMasks):
     rng = np.random.default_rng()
     # s = rng.binomial(1, probaBubbles, nFeatures)
     # print(s)
     # print(s.shape)
     # vec = masks[iSample,:]
     # vec[:] = 0
     # bubblePos = (np.floor(np.random.rand(1,nbBubbles)*nFeatures)).astype(int)
     # vec[bubblePos] = 1   
     vec = rng.binomial(1, probaBubbles, nFeatures).astype(float)
     vec = np.reshape(vec, dimOfinput) 
     binaryMasks[iSample,:] = vec.flatten()
     vec = gaussian_filter(vec, sigma=bubbleSize)
     
     masks[iSample,:] = vec.flatten()
     masks[iSample,:] /= np.amax(masks[iSample,:])
     masks[iSample,:] -= np.amin(masks[iSample,:])
     masks[iSample,:] /= np.amax(masks[iSample,:])
   return masks, binaryMasks

# generate probing samples from training set or with noise, or with pseudo-random noise from a testing set
def generateProbingSamples(x_train_set = [], x_test_set = [], dimOfinput=(28,28), bubbleMasks = [], probingMethod = 'bubbles', samplesMethod = 'gaussianNoise', nDim_pca = 50, nbRevcorTrials = 1000, normalizeNoiseCoeff = 10):
   noise = []
   if probingMethod == 'bubbles':
     N_probing_samples = bubbleMasks.shape[0]
     if samplesMethod == 'pseudoRandom':
       pca_noise = PCA(n_components= nDim_pca, svd_solver='auto', whiten=True).fit(np.reshape(x_test_set,(x_test_set.shape[0],np.product(dimOfinput))))
       std_pca_estimated = np.std(pca_noise.transform(np.reshape(x_test_set,(x_test_set.shape[0],np.product(dimOfinput)))).flatten())
       probingSamples = pca_noise.inverse_transform(np.random.randn(bubbleMasks.shape[0],nDim_pca)*std_pca_estimated) * bubbleMasks
     elif samplesMethod == 'gaussianNoise':
       probingSamples = np.random.randn(bubbleMasks.shape[0],bubbleMasks.shape[1]) * bubbleMasks
     elif samplesMethod == 'trainSet':
       probingSamples =  np.reshape(x_train_set,(x_train_set.shape[0],np.product(dimOfinput)))[0:bubbleMasks.shape[0],:] * bubbleMasks
     elif samplesMethod == 'testSet':
       probingSamples =  np.reshape(x_test_set,(x_test_set.shape[0],np.product(dimOfinput)))[0:bubbleMasks.shape[0],:] * bubbleMasks       
   elif probingMethod == 'revcor':
     if samplesMethod == 'pseudoRandom':
       pca_noise = PCA(n_components= nDim_pca, svd_solver='auto', whiten=True).fit(np.reshape(x_test_set,(x_test_set.shape[0],np.product(dimOfinput))))
       std_pca_estimated = np.std(pca_noise.transform(np.reshape(x_test_set,(x_test_set.shape[0],np.product(dimOfinput)))).flatten())
       noise_pca = np.random.randn(nbRevcorTrials,nDim_pca)*std_pca_estimated
       probingSamples = pca_noise.inverse_transform(noise_pca)
       noise = noise_pca
     elif samplesMethod == 'gaussianNoise':
       probingSamples = np.random.randn(nbRevcorTrials,np.product(dimOfinput)) 
     elif samplesMethod == 'trainSet':
       # noise = np.random.randn(nbRevcorTrials,np.product(dimOfinput)) / normalizeNoiseCoeff 
       print('before')
       noise = np.random.randn(nbRevcorTrials,np.product(dimOfinput)) / normalizeNoiseCoeff
       #noise = np.reshape(generate_perlin_noise_3d((nbRevcorTrials, 28, 28), (nbRevcorTrials, 20, 20), tileable=(True, False, False)),(nbRevcorTrials,28*28))
       # n1 = 4
       # n2 = 4  
       # n3 = 7
       # n4 = 7
       # n5 = 7
       # noise1 = np.reshape(generate_fractal_noise_3d((int(nbRevcorTrials/10), 28, 28), (int(nbRevcorTrials/10), n1, n1), tileable=(True, False, False)),(int(nbRevcorTrials/10),28*28)) * normalizeNoiseCoeff
       # noise2 = np.reshape(generate_fractal_noise_3d((int(nbRevcorTrials/10), 28, 28), (int(nbRevcorTrials/10), n2, n2), tileable=(True, False, False)),(int(nbRevcorTrials/10),28*28)) * normalizeNoiseCoeff
       # noise3 = np.reshape(generate_fractal_noise_3d((int(nbRevcorTrials/10), 28, 28), (int(nbRevcorTrials/10), n3, n3), tileable=(True, False, False)),(int(nbRevcorTrials/10),28*28)) * normalizeNoiseCoeff
       # noise4 = np.reshape(generate_fractal_noise_3d((int(nbRevcorTrials/10), 28, 28), (int(nbRevcorTrials/10), n4, n5), tileable=(True, False, False)),(int(nbRevcorTrials/10),28*28)) * normalizeNoiseCoeff
       # noise5 = np.reshape(generate_fractal_noise_3d((int(nbRevcorTrials/10), 28, 28), (int(nbRevcorTrials/10), n5, n5), tileable=(True, False, False)),(int(nbRevcorTrials/10),28*28)) * normalizeNoiseCoeff
       # noise6 = np.reshape(generate_fractal_noise_3d((int(nbRevcorTrials/10), 28, 28), (int(nbRevcorTrials/10), n1, n1), tileable=(True, False, False)),(int(nbRevcorTrials/10),28*28)) * normalizeNoiseCoeff
       # noise7 = np.reshape(generate_fractal_noise_3d((int(nbRevcorTrials/10), 28, 28), (int(nbRevcorTrials/10), n2, n2), tileable=(True, False, False)),(int(nbRevcorTrials/10),28*28)) * normalizeNoiseCoeff
       # noise8 = np.reshape(generate_fractal_noise_3d((int(nbRevcorTrials/10), 28, 28), (int(nbRevcorTrials/10), n3, n3), tileable=(True, False, False)),(int(nbRevcorTrials/10),28*28)) * normalizeNoiseCoeff
       # noise9 = np.reshape(generate_fractal_noise_3d((int(nbRevcorTrials/10), 28, 28), (int(nbRevcorTrials/10), n4, n4), tileable=(True, False, False)),(int(nbRevcorTrials/10),28*28)) * normalizeNoiseCoeff
       # noise10 = np.reshape(generate_fractal_noise_3d((int(nbRevcorTrials/10), 28, 28), (int(nbRevcorTrials/10), n5, n5), tileable=(True, False, False)),(int(nbRevcorTrials/10),28*28)) * normalizeNoiseCoeff
       
       # noise = np.concatenate((noise1,noise2,noise3,noise4,noise5,noise6,noise7,noise8,noise9,noise10),axis=0)
       
       print(noise.shape)
       # HERE GOOD ONE FOR MNIST
       # noise = np.reshape(gabor(N=28,Nit=10,Nnoise=nbRevcorTrials),(nbRevcorTrials,np.product(dimOfinput))) * normalizeNoiseCoeff

       # plt.imshow(np.reshape(noise[20,:],dimOfinput))
       # plt.show()
       # pca_noise = PCA(n_components= nDim_pca, svd_solver='auto', whiten=True).fit(np.reshape(np.random.randn(nbRevcorTrials,np.product(dimOfinput)),(np.random.rand(nbRevcorTrials,np.product(dimOfinput)).shape[0],np.product(dimOfinput))))
       # std_pca_estimated = np.std(pca_noise.transform(np.reshape(x_test_set,(x_test_set.shape[0],np.product(dimOfinput)))).flatten())
       # noise_pca = np.random.randn(nbRevcorTrials,nDim_pca)*std_pca_estimated
       # noise = pca_noise.inverse_transform(noise_pca)
       probingSamples = np.reshape(x_test_set,(x_test_set.shape[0],np.product(dimOfinput)))[0:nbRevcorTrials,:] + noise 
       print('after')
       # plt.plot(probingSamples[10,:])
       # plt.show()
       # probingSamples = noise
       # probingSamples[probingSamples>1]=1
       # probingSamples[probingSamples<0]=0
       # probingSamples -= np.min(probingSamples)
       # probingSamples /= np.max(probingSamples)
   return probingSamples, noise


# # Optimize the parameter ProbaBubbles in order to have ~75% of success
# def OptimizeProbaBubbles(model,categs):
   
#   y_pred = model.predict_classes(probingOneSamples)
   
#   return tabMaps


