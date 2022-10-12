# from sklearnex import patch_sklearn
# patch_sklearn(["SVC"])

from timeit import default_timer as timer
import numpy as np 
import glob 
import pickle
import matplotlib.pyplot as plt  
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict, GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score
from sklearn.decomposition import PCA
from scipy import stats
import scipy.io as sio
from lib import proise_v2
import functions
# from numba import jit
# import pingouin as pg

# @jit(nopython=True)
fileList = glob.glob("./stmtf/*.pkl") #stmtf
tabStrf  = []
tabSession = []
tabDaySession = []
tabSubjectNb = []
tabSegment = []
tabFilename = []
outFolder = './out/out_03_classicationSubjectLevel_AllMaps/'
dataFolder = './data/'

#load data
data = pickle.load(open(str(dataFolder)+'data.pkl', 'rb'))

tabStrf = data['tabStrf']
tabStrf = tabStrf / np.amax(tabStrf)
tabSession = data['tabSession']
# tabDaySession = data['daySession']
tabSubjectNb = data['tabSubjectNb']
del data
tabMeanAccuracy = []
tabStdAccuracy = []


# print(dir())

tabStrf = np.asarray(tabStrf)
tabSession = np.asarray(tabSession)
tabSubjectNb = np.asarray(tabSubjectNb)
subjectNbTab = np.unique(tabSubjectNb) # unique subject tab
tabDaySession = np.asarray(tabDaySession)


# label to classify
class_ = tabSession
# print(class_)

# PCA on data
n_components_tab = [3, 15, 30, 50, 100, 150, 200, 250]
n_components_tab = [70]
scoreMeanTab = []
scoreStdTab  = []
pca_tab = []
clf_tab = []

cv = StratifiedKFold(5, shuffle=True)

# tabStrf = stats.zscore(tabStrf, axis=0)
tabScaleRate = []
projection = 1
probingSamplesProjection = []
nbChannels = 128
nbRates = 22
nbScales = 8
print("n component")

# for n_components in n_components_tab:
#   # print(str(n_components))
#   pca2store = PCA(n_components=n_components, svd_solver='auto', whiten=True).fit(tabStrf)
#   print(np.cumsum(pca2store.explained_variance_ratio_))
#   pca_tab.append(pca2store)
#   exp_ratio = np.cumsum(pca2store.explained_variance_ratio_)
#   exp_ratio[exp_ratio<.99] = 1000
#   nDim_optimal_pca = np.argmin(exp_ratio)
#   print('nb optimal dims :'+str(nDim_optimal_pca))  

  # print('explained variance :'+str(np.cumsum(pca2store.explained_variance_ratio_)))
  # exp_ratio = np.cumsum(pca2store.explained_variance_ratio_)
  # exp_ratio[exp_ratio<.9] = 1000

  # print(np.argmin(exp_ratio))
tabBAcc = []
nDim_optimal_pca_tab = []
Ntimes = 1

for iSubject in subjectNbTab:
  tabInterpBySubject = []
  tabBAccTemp = []
  for iRepeat in range(Ntimes):
    print('Subject #',str(iSubject),' Repeat #',str(iRepeat))
    X = []
    Y = []
    X = tabStrf[tabSubjectNb==(iSubject)][:]
    Y = class_[tabSubjectNb==(iSubject)]
    pca_optimal_dim = PCA(n_components=70, svd_solver='auto', whiten=True).fit(tabStrf[tabSubjectNb!=(iSubject)][:])
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=iRepeat)
    del X
    del Y
    print(X_train.shape)
    # n_components = X_train.shape[0]
    # pca = PCA(n_components=n_components, svd_solver='auto', whiten=True).fit(X_train)
    # print('explained variance :'+str(np.sum(pca.explained_variance_ratio_)))
    # exp_ratio = np.cumsum(pca.explained_variance_ratio_)
    # exp_ratio[exp_ratio<.99] = 1000
    # nDim_optimal_pca = np.argmin(exp_ratio)
    # print('nb optimal dims :'+str(nDim_optimal_pca))
    # nDim_optimal_pca_tab.append(nDim_optimal_pca)
    # pca_optimal_dim = PCA(n_components=nDim_optimal_pca, svd_solver='auto', whiten=True).fit(X_train)
    # print('explained variance optimal dim :'+str(np.sum(pca_optimal_dim.explained_variance_ratio_)))

    # for n_components in n_components_tab:
    for count, n_components in enumerate(n_components_tab):
      # pca_optimal_dim = PCA(n_components=n_components, svd_solver='auto', whiten=True).fit(X_train)
      print('explained variance :'+str(np.sum(pca_optimal_dim.explained_variance_ratio_)))
      start = timer()      
      X_pca_train = pca_optimal_dim.transform(X_train)
      # grid search classifier definition and fit
      tuned_parameters = { 'gamma':np.logspace(-3,3,num=5),'C':np.logspace(-3,3,num=5)}
      clf = GridSearchCV(SVC(kernel='rbf'), tuned_parameters, n_jobs=15, cv=cv,  pre_dispatch=6,
                     scoring='balanced_accuracy', verbose=False)
      clf.fit(X_pca_train, Y_train)
      clf_tab.append(clf)
      scoreMeanTab.append(clf.cv_results_['mean_test_score'][clf.best_index_])
      scoreStdTab.append(clf.cv_results_['std_test_score'][clf.best_index_])
      # classifier 10-fold cross-validation
      # scores = cross_val_score(SVC(kernel='rbf',gamma=clf.best_params_['gamma'], C=clf.best_params_['C']), X_pca_train, Y_train, cv=cv)
      Y_test_pred = clf.predict(pca_optimal_dim.transform(X_test))

      # print(balanced_accuracy_score(Y_test,Y_test_pred))
      tabBAccTemp.append(balanced_accuracy_score(Y_test,Y_test_pred))
      # interpretation stim+pseudo-noise
      N = 100
      dimOfinput = (128,8,22) # dimension of the input representation
      probingMethod = 'revcor' # choice of the probing method : bubbles or revcor
      samplesMethod = 'pseudoRandom' # choice of the method to generate probing samples trainSet or pseudoRandom (need x_test_set) or gaussianNoise
      nbRevcorTrials = X_train.shape[0]*N # number of probing samples for the reverse correlation (must be below the number of training sample if trainSet)
      nDim_pca = 30 # number of dimension to compute the PCA for the pseudo-random noise generation

      probingSamples, _ = proise_v2.generateProbingSamples(x_train_set = X_train, x_test_set = X_train, dimOfinput=dimOfinput, probingMethod = probingMethod, samplesMethod = samplesMethod, nDim_pca = nDim_pca, nbRevcorTrials = nbRevcorTrials)
      data2transform = probingSamples/2 + np.tile(X_train,(N,1))/2
      print(data2transform.shape)
      # X_probingSamples_pca = pca_optimal_dim.transform(probingSamples/2 + np.tile(X_train,(N,1))/2)  

      X_probingSamples_pca_1 = pca_optimal_dim.transform(data2transform[0:int(data2transform.shape[0]/2),:])
      y_pred = clf.predict(X_probingSamples_pca_1)
      del X_probingSamples_pca_1
      X_probingSamples_pca_2 = pca_optimal_dim.transform(data2transform[int(data2transform.shape[0]/2):data2transform.shape[0],:])        
      y_pred = np.concatenate((y_pred,clf.predict(X_probingSamples_pca_2)))
      del X_probingSamples_pca_2
      # X_probingSamples_pca = np.concatenate((X_probingSamples_pca_1,X_probingSamples_pca_1),axis=0)
      # y_pred = clf.predict(X_probingSamples_pca)

      responses_ = np.squeeze((y_pred == np.tile(Y_train,(1,N))) & (np.tile(Y_train,(1,N)) == 'post'))
      canonicalMap = []

      data2revcor = np.asarray(probingSamples)
      canonicalMap = np.nanmean(data2revcor[responses_][:],axis=0) - np.nanmean(data2revcor[np.logical_not(responses_)][:],axis=0)

      print(np.mean(probingSamples[responses_][:],axis=0).shape)
      # canonicalMap_ToPlot = np.asarray(canonicalMap)
      tabInterpBySubject.append(canonicalMap)
      del canonicalMap
      end = timer()
      print('Elapsed time:'+str(end-start))
  sio.savemat(outFolder+str(iSubject).zfill(3)+'_AllMaps_3d.mat', {'canonicalAllMaps': np.asarray(tabInterpBySubject), 'iSubject': iSubject, 'nDim_optimal_pca_tab':np.asarray(nDim_optimal_pca_tab), 'explained_variance':np.sum(pca_optimal_dim.explained_variance_ratio_)})
  # sio.savemat(outFolder+str(iSubject).zfill(3)+'_masks_3d.mat', {'canonicalMap': np.mean(canonicalMap), 'canonicalAllMaps': np.asarray(tabInterpBySubject), 'iSubject': iSubject})
  tabBAcc.append(tabBAccTemp)
  print(np.mean(np.asarray(tabBAccTemp)))
  print(np.std(np.asarray(tabBAccTemp)))

sio.savemat(outFolder+'BAcc_3D.mat', {'tabBAcc_3d': np.asarray(tabBAcc)})        


