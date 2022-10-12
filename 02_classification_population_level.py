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

# projection = 1 ;
# for n_stim in range(tabStrf.shape[0]):
#   strf_scale_rate, strf_freq_rate, strf_freq_scale = functions.avgvec2strfavg(functions.strf2avgvec(np.reshape(tabStrf[n_stim,:],(128,22,8)), abs_=True),nbChannels=128,nbRates=22,nbScales=8)
#   if projection == 1:
#     tabStrfProjection.append(strf_scale_rate.flatten())
#   elif projection == 2:
#     strf_freq_rate = scipy.signal.decimate(strf_freq_rate, 3, axis=0)
#     tabStrfProjection.append(strf_freq_rate.flatten())
#   elif projection == 3:
#     strf_freq_scale = scipy.signal.decimate(strf_freq_scale, 3, axis=0)    
#     tabStrfProjection.append(strf_freq_scale.flatten())

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

  # classifier k-fold cross-validation
  # scores = cross_val_score(SVC(kernel='rbf',gamma=clf.best_params_['gamma'], C=clf.best_params_['C']), X_train_pca, y_train, cv=cv)
  # print("Training accuracy (5-fold CV): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
  # trainScoresMean.append(scores.mean())
  # trainScoresStd.append(scores.std())
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

  # printFigScaleRate(X_train[12,:],figname='sample12.png')
  # printFigScaleRate(X_train[100,:],figname='sample100.png')
  # printFigScaleRate(X_train[200,:],figname='sample200.png')
  # printFigScaleRate(X_train[1000,:],figname='sample1000.png')
  # printFigScaleRate(X_train[300,:],figname='sample300.png')
  # printFigScaleRate(X_train[456,:],figname='sample456.png')

  # printFigScaleRate(probingSamples[12,:],figname='probing12.png')
  # printFigScaleRate(probingSamples[100,:],figname='probing100.png')
  # printFigScaleRate(probingSamples[200,:],figname='probing200.png')
  # printFigScaleRate(probingSamples[1000,:],figname='probing1000.png')
  # printFigScaleRate(probingSamples[300,:],figname='probing300.png')
  # printFigScaleRate(probingSamples[456,:],figname='probing456.png')

  # X_probingSamples_pca = pca_tab[count].transform(probingSamples)      
  # y_pred = clf.predict(X_probingSamples_pca)
  # responses_ = y_pred == 'post'
  # print(confusion_matrix(np.tile(Y_train,(1,N)),y_pred))

  canonicalMap = []
  pval = []

  data2revcor = np.asarray(probingSamples)

  canonicalMap = np.nanmean(data2revcor[responses_][:],axis=0) - np.nanmean(data2revcor[np.logical_not(responses_)][:],axis=0)
  canonicalAllMaps.append(canonicalMap)
  # canonicalMap[np.abs(canonicalMap)<np.mean(np.abs(canonicalMap))] = np.nan
  # print(np.mean(probingSamples[responses_][:],axis=0).shape)

  # for iFeature in range(128*8*22):
  #   corr_ = pg.corr(probingSamples[:,iFeature], responses_, method='pearson')
  #   canonicalMap.append(corr_['r'])
  #   pval.append(corr_['p-val'])
  # pval_toTest = np.asarray(pval)
  # canonicalMap_ToPlot = np.asarray(canonicalMap)
  sio.savemat(outFolder+'BetweenSubjects_AllMaps_3d.mat', {'canonicalAllMaps': np.asarray(canonicalAllMaps),'explained_varianceTab':np.asarray(explained_varianceTab), 'tabBAcc':np.asarray(tabBAcc)})

  # printFigScaleRate(canonicalMap_ToPlot,figname='tttt.png')


  ################################################################################################################################

  # toPlot = np.reshape(canonicalMap_ToPlot,(8,22))
  # bound_value = np.max([np.abs(np.nanmin(toPlot.flatten())),np.abs(np.nanmax(toPlot.flatten()))])
  # # bound_value = .001
  # # plt.suptitle(title, fontsize=10)
  # fig, ax = plt.subplots(nrows=1, ncols=2)

  # # plt.plot(canonicalMap_ToPlot.flatten())
  # # plt.show()
  # # plt.imshow(np.reshape(canonicalMap_ToPlot,(8,22)))
  # # plt.show()
  # sio.savemat(outFolder+'folds'+str(fold).zfill(3)+'_masks.mat', {'canonicalMap': canonicalMap, 'iFold': fold})
  # plt.subplot(1,2,1)
  # im = plt.imshow(toPlot[:,0:11], aspect='equal', interpolation='hamming',origin='lower',cmap='jet', vmin=-bound_value, vmax = bound_value)
  # # fig.colorbar(im, aspect=1.3, format='%.2f', ticks=[0,bound_value])  
  # plt.xticks([])
  # plt.yticks([])
  # plt.axis('off')
  # plt.tight_layout()     

  # plt.subplot(1,2,2)
  # im = plt.imshow(toPlot[:,11:22], aspect='equal', interpolation='hamming',origin='lower',cmap='jet', vmin=-bound_value, vmax = bound_value)
  # # fig.colorbar(im, aspect=1.3, format='%.2f', ticks=[0,bound_value])  
  # plt.xticks([])
  # plt.yticks([])
  # plt.axis('off')
  # plt.tight_layout()
  # plt.subplots_adjust(wspace=0.15, hspace=0)  
  # plt.savefig('populationLevel_sujectMap'+'.png',bbox_inches='tight')     
  # # plt.show()







  # # interpretation stim+pseudo-noise
  # dimOfinput = (128,8,22) # dimension of the input representation
  # probingMethod = 'revcor' # choice of the probing method : bubbles or revcor
  # samplesMethod = 'pseudoRandom' # choice of the method to generate probing samples trainSet or pseudoRandom (need x_test_set) or gaussianNoise
  # nbRevcorTrials = X_train.shape[0] # number of probing samples for the reverse correlation (must be below the number of training sample if trainSet)
  # nDim_pca = 100 # number of dimension to compute the PCA for the pseudo-random noise generation
  # probingSamples, fff = proise_v2.generateProbingSamples(x_train_set = tabStrf, x_test_set = tabStrf, dimOfinput=dimOfinput, probingMethod = probingMethod, samplesMethod = samplesMethod, nDim_pca = nDim_pca, nbRevcorTrials = nbRevcorTrials)
  # print(probingSamples.shape)
  # X_probingSamples_pca = pca.transform(probingSamples + X_train)  
  # y_pred = clf.predict(X_probingSamples_pca)
  # print(confusion_matrix(y_train,y_pred))
  # # print(y_pred)
  # responses_ = (y_pred == y_train)
  # canonicalMap = []
  # pval = []
  # for iFeature in range(128*8*22):
  #   corr_ = pg.corr(probingSamples[:,iFeature], responses_, method='pearson')
  #   canonicalMap.append(corr_['r'])
  #   pval.append(corr_['p-val'])
  # pval_toTest = np.asarray(pval)
  # canonicalMap_ToPlot = np.asarray(canonicalMap)
  # #canonicalMap_ToPlot = (np.asarray(pval_toTest)[0,:]<.05/(128*8*22))*np.asarray(canonicalMap_ToPlot)
  # strf_scale_rate, strf_freq_rate, strf_freq_scale = functions.avgvec2strfavg(functions.strf2avgvec(np.reshape(canonicalMap_ToPlot,(128,8,22)), abs_=False),nbChannels=128,nbRates=22,nbScales=8)
  # functions.plotStrfavgEqual(strf_scale_rate, strf_freq_rate, strf_freq_scale,cmap='coolwarm',figname='pop_level.png')    

  # # canonicalMaps = proise_v2.ComputeCanonicalMaps(probingSamples,y_pred, dimOfinput=dimOfinput)
  # # canonicalMaps = proise_v2.ComputeCanonicalMaps(probingSamples, y_pred, dimOfinput=dimOfinput)
  # # canonicalMaps_pca = proise_v2.ComputeCanonicalMaps(X_probingSamples_pca, y_pred, dimOfinput=nDim_pca)
  # # canonicalMaps = pca.inverse_transform(canonicalMaps_pca)
  # # canonicalMaps, tabCI95, tabPval, tabZscore = proise_v2.ComputeCanonicalMapsCorrelation(probingSamples, y_pred, dimOfinput=dimOfinput, max_map=1)
  # canonicalMaps = proise_v2.ComputeCanonicalMapsWithGroundTruth(probingSamples,y_pred,y_train,dimOfinput)
  # # canonicalMaps = proise_v2.ComputeCanonicalMaps(probingSamples, y_pred, dimOfinput=dimOfinput)
  # toPlot = np.asarray(canonicalMaps)[0,:]
  # # toPlot = (np.asarray(tabPval)[0,:]<.05/(128*8*22))*np.asarray(canonicalMaps)[0,:]   
  # # toPlot[toPlot==0] = np.nan  
  # print(np.asarray(canonicalMaps).shape)
  # strf_scale_rate, strf_freq_rate, strf_freq_scale = functions.avgvec2strfavg(functions.strf2avgvec(np.reshape(toPlot,(128,8,22)), abs_=False),nbChannels=128,nbRates=22,nbScales=8)
  # functions.plotStrfavgEqual(strf_scale_rate, strf_freq_rate, strf_freq_scale,cmap='coolwarm',figname='pop_level.png')
  # toPlot = np.asarray(canonicalMaps)[1,:]
  # # toPlot = (np.asarray(tabPval)[0,:]<.05/(128*8*22))*np.asarray(canonicalMaps)[0,:]   
  # # toPlot[toPlot==0] = np.nan  
  # print(np.asarray(canonicalMaps).shape)
  # strf_scale_rate, strf_freq_rate, strf_freq_scale = functions.avgvec2strfavg(functions.strf2avgvec(np.reshape(toPlot,(128,8,22)), abs_=False),nbChannels=128,nbRates=22,nbScales=8)
  # functions.plotStrfavgEqual(strf_scale_rate, strf_freq_rate, strf_freq_scale,cmap='coolwarm',figname='pop_level.png')
  # toPlot = np.asarray(canonicalMaps)[0,:]-np.asarray(canonicalMaps)[1,:]
  # # toPlot = (np.asarray(tabPval)[0,:]<.05/(128*8*22))*np.asarray(canonicalMaps)[0,:]   
  # # toPlot[toPlot==0] = np.nan  
  # print(np.asarray(canonicalMaps).shape)
  # strf_scale_rate, strf_freq_rate, strf_freq_scale = functions.avgvec2strfavg(functions.strf2avgvec(np.reshape(toPlot,(128,8,22)), abs_=False),nbChannels=128,nbRates=22,nbScales=8)
  # functions.plotStrfavgEqual(strf_scale_rate, strf_freq_rate, strf_freq_scale,cmap='coolwarm',figname='pop_level.png')
  # toPlot = np.asarray(canonicalMaps)[1,:]-np.asarray(canonicalMaps)[0,:]


  # # interpretation pseudo-noise alone
  # dimOfinput = (128,8,22) # dimension of the input representation
  # probingMethod = 'revcor' # choice of the probing method : bubbles or revcor
  # samplesMethod = 'pseudoRandom' # choice of the method to generate probing samples trainSet or pseudoRandom (need x_test_set) or gaussianNoise
  # nbRevcorTrials = 1000 # number of probing samples for the reverse correlation (must be below the number of training sample if trainSet)
  # nDim_pca = 100 # number of dimension to compute the PCA for the pseudo-random noise generation
  # probingSamples, fff = proise_v2.generateProbingSamples(x_train_set = tabStrf, x_test_set = tabStrf, dimOfinput=dimOfinput, probingMethod = probingMethod, samplesMethod = samplesMethod, nDim_pca = nDim_pca, nbRevcorTrials = nbRevcorTrials)
  # print(probingSamples.shape)
  # X_probingSamples_pca = pca.transform(probingSamples)
  # y_pred = clf.predict(X_probingSamples_pca)
  # # print(y_pred)

  # # canonicalMaps = proise_v2.ComputeCanonicalMaps(probingSamples,y_pred, dimOfinput=dimOfinput)
  # # canonicalMaps = proise_v2.ComputeCanonicalMaps(probingSamples, y_pred, dimOfinput=dimOfinput)
  # # canonicalMaps_pca = proise_v2.ComputeCanonicalMaps(X_probingSamples_pca, y_pred, dimOfinput=nDim_pca)
  # # canonicalMaps = pca.inverse_transform(canonicalMaps_pca)
  # # canonicalMaps, tabCI95, tabPval, tabZscore = proise_v2.ComputeCanonicalMapsCorrelation(probingSamples, y_pred, dimOfinput=dimOfinput, max_map=1)
  # canonicalMaps = proise_v2.ComputeCanonicalMaps(probingSamples, y_pred, dimOfinput=dimOfinput)
  # toPlot = np.asarray(canonicalMaps)[0,:]
  # # toPlot = (np.asarray(tabPval)[0,:]<.05/(128*8*22))*np.asarray(canonicalMaps)[0,:]   
  # # toPlot[toPlot==0] = np.nan  
  # strf_scale_rate, strf_freq_rate, strf_freq_scale = functions.avgvec2strfavg(functions.strf2avgvec(np.reshape(toPlot,(128,8,22)), abs_=False),nbChannels=128,nbRates=22,nbScales=8)
  # functions.plotStrfavgEqual(strf_scale_rate, strf_freq_rate, strf_freq_scale,cmap='coolwarm',figname='pop_level.png')

