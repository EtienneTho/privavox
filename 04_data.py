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

  # exp_ratio[exp_ratio<.9] = 1000

  # print(np.argmin(exp_ratio))
tabBAcc = []
nDim_optimal_pca_tab = []
Ntimes = 1
nbSample_subjects = []
nbSample_train_subjects = []

for iSubject in subjectNbTab:
  tabInterpBySubject = []
  tabBAccTemp = []
  X = tabStrf[tabSubjectNb==(iSubject)][:]
  Y = class_[tabSubjectNb==(iSubject)]
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=1)
  print('Subject #',str(iSubject),' nb samples whole = ',str(X.shape[0]), ' ; nb_sample_train = ',str(X_train.shape[0]) )
  nbSample_subjects.append(X.shape[0])
  nbSample_train_subjects.append(X_train.shape[0])

print('Nb samples whole : M=',str(np.mean(nbSample_subjects)),' min=',str(np.min(nbSample_subjects)),' max=',str(np.max(nbSample_subjects)))
print('Nb samples train : M=',str(np.mean(nbSample_train_subjects)),' min=',str(np.min(nbSample_train_subjects)),' max=',str(np.max(nbSample_train_subjects)))


    # sio.savemat(outFolder+str(iSubject).zfill(3)+'_AllMaps_3d.mat', {'canonicalAllMaps': np.asarray(tabInterpBySubject), 'iSubject': iSubject, 'nDim_optimal_pca_tab':np.asarray(nDim_optimal_pca_tab), 'explained_variance':np.sum(pca_optimal_dim.explained_variance_ratio_)})


