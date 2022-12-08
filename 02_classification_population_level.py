import numpy as np 
import glob 
import pickle
import matplotlib.pyplot as plt  
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict, GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score
from sklearn.decomposition import PCA, FastICA
from scipy import stats
import scipy.io as sio
from lib import proise_v2
import functions
import pingouin as pg

def printFigScaleRate(canonicalMap_ToPlot,figname='default.png'):
  toPlot = np.reshape(canonicalMap_ToPlot,(8,22))
  bound_value = np.max([np.abs(np.nanmin(toPlot.flatten())),np.abs(np.nanmax(toPlot.flatten()))])
  # bound_value = .001
  # plt.suptitle(title, fontsize=10)
  fig, ax = plt.subplots(nrows=1, ncols=2)

  # plt.plot(canonicalMap_ToPlot.flatten())
  # plt.show()
  # plt.imshow(np.reshape(canonicalMap_ToPlot,(8,22)))
  # plt.show()
  plt.subplot(1,2,1)
  im = plt.imshow(toPlot[:,0:11], aspect='equal', interpolation='hamming',origin='lower',cmap='jet', vmin=-bound_value, vmax = bound_value)
  # fig.colorbar(im, aspect=1.3, format='%.2f', ticks=[0,bound_value])  
  plt.xticks([])
  plt.yticks([])
  plt.axis('off')
  plt.tight_layout()     

  plt.subplot(1,2,2)
  im = plt.imshow(toPlot[:,11:22], aspect='equal', interpolation='hamming',origin='lower',cmap='jet', vmin=-bound_value, vmax = bound_value)
  # fig.colorbar(im, aspect=1.3, format='%.2f', ticks=[0,bound_value])  
  plt.xticks([])
  plt.yticks([])
  plt.axis('off')
  plt.tight_layout()
  plt.subplots_adjust(wspace=0.15, hspace=0)  
  plt.savefig(figname,bbox_inches='tight')
  # plt.show()


fileList = glob.glob("./stmtf/*.pkl") #stmtf
tabStrf  = []
tabSession = []
tabDaySession = []
tabSubjectNb = []
tabSegment = []
tabFilename = []
tabStrfProjection = []

outFolder = './out/out_02_classication_population_Level_AllMaps/'
dataFolder = './data/'

#load data
data = pickle.load(open(str(dataFolder)+'data.pkl', 'rb'))
# print(data)

tabStrf = data['tabStrf']
tabStrf = tabStrf / np.amax(tabStrf)
tabSession = data['tabSession']
# tabDaySession = data['daySession']
tabSubjectNb = data['tabSubjectNb']
tabMeanAccuracy = []
tabStdAccuracy = []

tabStrf = np.asarray(tabStrf)
tabSession = np.asarray(tabSession)
tabSubjectNb = np.asarray(tabSubjectNb)
tabDaySession = np.asarray(tabDaySession)

print(np.unique(tabDaySession))

# label to classify
class_ = tabSession
print(class_)

# PCA on data
n_components_tab = [3, 10, 30, 50, 100, 150, 200, 250]
n_components = 250
scoreMeanTab = []
scoreStdTab  = []

n_fold = 50
n_cv = 5
cv = StratifiedKFold(n_cv, shuffle=True)
trainScoresMean = []
trainScoresStd  = []
testScore       = []
tabBAcc         = []

tabStrf = np.asarray(tabStrf)
print(tabStrf.shape)

canonicalAllMaps = []
explained_varianceTab = []

for fold in range(n_fold):
  X_train, X_test, y_train, y_test = train_test_split(tabStrf, class_, test_size=0.25, random_state=fold)
  print(X_train.shape)

  pca = PCA(n_components=n_components, svd_solver='auto', whiten=True).fit(X_train)   
  X_train_pca = pca.transform(np.asarray(X_train))
  print('explained variance :'+str(np.sum(pca.explained_variance_ratio_)))
  explained_varianceTab.append(np.sum(pca.explained_variance_ratio_))
  print('PCA done '+str(n_components))

  # grid search classifier definition and fit
  tuned_parameters = { 'gamma':np.logspace(-3,3,num=3),'C':np.logspace(-3,3,num=3)}
  clf = GridSearchCV(SVC(kernel='rbf'), tuned_parameters, n_jobs=-1, cv=cv,  pre_dispatch=6,
                 scoring='balanced_accuracy', verbose=False)
  clf.fit(X_train_pca, y_train)
  print('GridSearch done')

  y_pred = clf.predict(pca.transform(X_test))
  testScore.append(balanced_accuracy_score(y_test,y_pred))
  print("testing accuracy: %0.2f" % (balanced_accuracy_score(y_test,y_pred)) )
  tabBAcc.append(balanced_accuracy_score(y_test,y_pred))

  ################################################################################################################################
  # interpretation stim+pseudo-noise
  N = 5
  dimOfinput = (128,8,22) # dimension of the input representation
  probingMethod = 'revcor' # choice of the probing method : bubbles or revcor
  samplesMethod = 'pseudoRandom' # choice of the method to generate probing samples trainSet or pseudoRandom (need x_test_set) or gaussianNoise
  nbRevcorTrials = X_train.shape[0]*N # number of probing samples for the reverse correlation (must be below the number of training sample if trainSet)
  nDim_pca = 30 # number of dimension to compute the PCA for the pseudo-random noise generation
  probingSamples, _ = proise_v2.generateProbingSamples(x_train_set = X_train, x_test_set = X_train, dimOfinput=dimOfinput, probingMethod = probingMethod, samplesMethod = samplesMethod, nDim_pca = nDim_pca, nbRevcorTrials = nbRevcorTrials)
  print(probingSamples.shape)
  X_probingSamples_pca = pca.transform(probingSamples/2 + np.tile(X_train,(N,1))/2)  
  y_pred = clf.predict(X_probingSamples_pca)
  responses_ = np.squeeze((y_pred == np.tile(y_train,(1,N))) & (np.tile(y_train,(1,N)) == 'post'))

  canonicalMap = []
  pval = []

  data2revcor = np.asarray(probingSamples)

  canonicalMap = np.nanmean(data2revcor[responses_][:],axis=0) - np.nanmean(data2revcor[np.logical_not(responses_)][:],axis=0)
  canonicalAllMaps.append(canonicalMap)

  sio.savemat(outFolder+'BetweenSubjects_AllMaps_3d.mat', {'canonicalAllMaps': np.asarray(canonicalAllMaps),'explained_varianceTab':np.asarray(explained_varianceTab), 'tabBAcc':np.asarray(tabBAcc)})