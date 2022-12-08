import numpy as np 
import glob 
import pickle
import functions
import matplotlib.pyplot as plt 

fileList = glob.glob("./stmtf/*.pkl") #stmtf
tabStrf  = []
tabRates = [] 
tabSession = []
tabDaySession = []
tabSubjectNb = []
tabSegment = []
tabStanford = []
tabFilename = []
tabSegment = []
tabSR = []
tabFR = []
tabFS = []
outFolder = './out_00_data/'


stanford = np.asarray([[1,3,3,1,1,1,1,3,1,2,1,3,4,3,2,3,1,1,3,1,1,1],
                        [1,1,3,2,1,2,1,1,1,1,1,1,3,3,2,4,2,3,4,1,2,2],
                        [1,2,1,3,1,3,1,1,1,1,1,3,4,5,2,2,2,3,3,1,3,2],
                        [5,3,5,1,3,5,2,5,2,2,2,2,2,3,2,5,3,3,3,3,3,3],
                        [3,3,6,2,2,6,4,3,6,1,3,3,6,3,2,4,4,4,4,2,4,2],
                        [2,2,6,1,3,5,4,2,3,3,2,3,6,6,3,5,3,5,4,2,6,3]])

subjectNumbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11, 12, 13, 14, 15, 15, 16, 17, 18, 19, 20, 21, 22]

# load data
for iFile, filename in enumerate(sorted(fileList)):
  print(str(iFile+1)+'/'+str(len(fileList)))
  dataFor = pickle.load(open(filename, 'rb'))
  tabFilename.append(filename)
  print(filename)
  # print(dataFor)
  toAdd = dataFor['strf']
  toAdd = toAdd / np.amax(toAdd)
  # toAdd = np.abs((np.fft.fft(toAdd)))
  # toAdd = np.log10(toAdd / np.amax(toAdd))
  print(toAdd.shape)
  tabStrf.append(toAdd.flatten())#/np.sum(dataFor['strf'].flatten())) # !!!!! NORMALISATION
  strf_scale_rate, strf_freq_rate, strf_freq_scale = functions.avgvec2strfavg(functions.strf2avgvec(toAdd, abs_=False),nbChannels=128,nbRates=22,nbScales=8)
  tabSR.append(strf_scale_rate.flatten())
  tabFR.append(strf_freq_rate.flatten())
  tabFS.append(strf_freq_scale.flatten())


  tabSession.append(dataFor['session'])
  tabSubjectNb.append(int(dataFor['subjectNb']))
  shift = 0
  daySession = dataFor['daySession']
  
  if dataFor['daySession'] == '02b':
    dataFor['daySession'] = '02'
  if dataFor['session'] == 'post': 
    shift = 3
  daySession = int(dataFor['daySession'])+int(shift)    
  tabDaySession.append(int(daySession))
  print()
  print(dataFor['durationSegment'])
  print(stanford.shape)
  print(np.asarray(subjectNumbers).shape)
  print(int(dataFor['subjectNb']))
  print(int(subjectNumbers[int(dataFor['subjectNb'])-1]))
  # print('subNb ' + str(int(subjectNumbers[int(dataFor['subjectNb'])])-1) + '_daySession ' + str(daySession-1))
  tabStanford.append(int(stanford[daySession-1][int(subjectNumbers[int(dataFor['subjectNb'])-1]-1)]))
  tabSegment.append(int(dataFor['iSegment']))

#save data
pickle.dump({'tabSR':tabSR, 'tabFR': tabFR, 'tabFS': tabFS, 'tabStrf': tabStrf, 'tabSegment': tabSegment, 'tabSession': tabSession, 'tabSubjectNb': tabSubjectNb, 'tabDaySession': tabDaySession, 'tabStanford': tabStanford}, open(str(outFolder)+'data.pkl', 'wb'))


