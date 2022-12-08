import numpy as np 
import glob 
import pickle
import matplotlib.pyplot as plt  
import functions
import scipy.io as sio

fileList = glob.glob("./stmtf/*.pkl")
tabStrf  = []
tabSession = []
tabDaySession = []
tabSubjectNb = []
tabFileNb = []
outFolder = './out/out_01_Acoustic_Analysis/'

dataFolder = './data/'

#load data
data = pickle.load(open(str(dataFolder)+'data.pkl', 'rb'))
tabStrf = data['tabStrf']
tabStrf = tabStrf 
tabSession = data['tabSession']
tabSubjectNb = data['tabSubjectNb']

tabStrf = np.asarray(tabStrf)
tabSession = np.asarray(tabSession)
tabSubjectNb = np.asarray(tabSubjectNb)
tabDaySession = np.asarray(tabDaySession)

projection = 1
probingSamplesProjection = []
nbChannels = 128
nbRates = 22
nbScales = 8

X = tabStrf
Y = tabSession
mean_tot  = np.mean(np.asarray(X),axis=0)
mean_pre  = np.mean(np.asarray(X[Y=='pre'][:]),axis=0)
mean_post = np.mean(np.asarray(X[Y=='post'][:]),axis=0)
diff_mean = np.divide(np.abs(mean_post-mean_pre),(mean_pre+mean_post)/2)
sio.savemat(outFolder+'AcousticAnalysis_BetweenSubjects.mat', {'mean_tot':np.asarray(mean_tot), 
                                                               'mean_pre': np.asarray(mean_pre), 
                                                               'mean_post': np.asarray(mean_post),
                                                               'diff_mean':  np.asarray(diff_mean)})

labels = np.zeros((Y.shape[0],1))
labels[Y=='pre']  = 0
labels[Y=='post'] = 1
sio.savemat(outFolder+'AcousticAnalysis_BetweenSubjects_all.mat', {'STMs':np.asarray(X),
                                                                   'labels':np.array(labels),
                                                                   'subjectNb':np.array(tabSubjectNb)})


# per subject
tabIndex = np.unique(tabSubjectNb)

for index in tabIndex:
  print('Subject nb '+str(index))
  X = []
  Y = []
  X = tabStrf[tabSubjectNb==(index)][:]
  Y = tabSession[tabSubjectNb==(index)]

  iSubject = index

  mean_pre = (np.mean(np.asarray(X[Y=='pre'][:]),axis=0))
  mean_post = (np.mean(np.asarray(X[Y=='post'][:]),axis=0))
  diff_mean = np.divide(np.abs(mean_post-mean_pre),(mean_pre+mean_post)/2)

  # print(outFolder+str(iSubject).zfill(3)+'_AllMaps_3d.mat')
  sio.savemat(outFolder+str(iSubject).zfill(3)+'AcousticAnalysis_sub.mat', {'mean_tot':np.asarray(mean_tot), 
                                                               'mean_pre': np.asarray(mean_pre), 
                                                               'mean_post': np.asarray(mean_post),
                                                               'diff_mean':  np.asarray(diff_mean),
                                                               'iSubject': np.asarray(iSubject)})
