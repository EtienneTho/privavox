import matplotlib.pyplot as plt  
import numpy as np 


def strf2avgvec(strf,timeDim=False, abs_=True):
    if abs_==True:
        if timeDim:
          strf_scale_rate = np.nanmean(np.abs(strf), axis=(0, 1))
          strf_freq_rate  = np.nanmean(np.abs(strf), axis=(0, 2))
          strf_freq_scale = np.nanmean(np.abs(strf), axis=(0, 3))
        else:
          strf_scale_rate = np.nanmean(np.abs(strf), axis=0)
          strf_freq_rate  = np.nanmean(np.abs(strf), axis=1)
          strf_freq_scale = np.nanmean(np.abs(strf), axis=2)         
    else:
        if timeDim:
          strf_scale_rate = np.nanmean((strf), axis=(0, 1))
          strf_freq_rate  = np.nanmean((strf), axis=(0, 2))
          strf_freq_scale = np.nanmean((strf), axis=(0, 3))
        else:
          strf_scale_rate = np.nanmean((strf), axis=0)
          strf_freq_rate  = np.nanmean((strf), axis=1)
          strf_freq_scale = np.nanmean((strf), axis=2)    
    avgvec = np.concatenate((np.ravel(strf_scale_rate), np.ravel(strf_freq_rate), np.ravel(strf_freq_scale)))
    return avgvec

def plotStrfavgEqualFixedBounds(strf_scale_rate, strf_freq_rate, strf_freq_scale,aspect_='equal', interpolation_='none',figname='defaut',show='true',title='',cmap='jet', minbounds = -1, maxbounds = 1):
    plt.suptitle(title, fontsize=10)
    # fig, ax = plt.subplots(nrows=1, ncols=3)
    plt.subplot(1,3,1)
    im = plt.imshow(strf_scale_rate, aspect=22/11, interpolation=interpolation_,origin='lower',cmap=cmap, vmin=-minbounds, vmax = maxbounds)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')   
    plt.tight_layout()        

    plt.subplot(1,3,2)
    im = plt.imshow(strf_scale_rate, aspect=22/11, interpolation=interpolation_,origin='lower',cmap=cmap, vmin=-minbounds, vmax = maxbounds)
    plt.xticks([])
    plt.yticks([]) 
    plt.axis('off')   
    plt.tight_layout()        

    # plt.xlabel('Rate (Hz)', fontsize=10)
    # plt.ylabel('Frequency (Hz)', fontsize=10)    
    plt.subplot(1,3,3)
    im = plt.imshow(strf_scale_rate, aspect=22/11, interpolation=interpolation_,origin='lower',cmap=cmap, vmin=-minbounds, vmax = maxbounds)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')   
    plt.tight_layout()        

    plt.savefig(figname+'.png',bbox_inches='tight')

    if show=='true':
        plt.show()     


def plotStrfavgEqual(strf_scale_rate, strf_freq_rate, strf_freq_scale,aspect_='equal', interpolation_='none',figname='defaut',show='true',title='',cmap='jet', minval = True):
    plt.suptitle(title, fontsize=10)
    # fig, ax = plt.subplots(nrows=1, ncols=3)
    plt.subplot(1,3,1)
    bound_value = np.max([np.abs(np.nanmin(strf_scale_rate.flatten())),np.abs(np.nanmax(strf_scale_rate.flatten()))])
    if minval == True:
        im = plt.imshow(strf_scale_rate, aspect=22/11, interpolation=interpolation_,origin='lower',cmap=cmap, vmin=-bound_value, vmax = bound_value)
        fig.colorbar(im, aspect=4, format='%.2f', ticks=[-bound_value,0,bound_value])
    else:
        im = plt.imshow(strf_scale_rate, aspect=22/11, interpolation=interpolation_,origin='lower',cmap=cmap, vmin=0, vmax = bound_value)
        fig.colorbar(im, aspect=4, format='%.2f', ticks=[0,bound_value])
    # plt.xlabel('Rate (Hz)', fontsize=10)
    # plt.ylabel('Scale (c/o)', fontsize=10)

    plt.xticks([])
    plt.yticks([])
    plt.axis('off')   
    plt.tight_layout()        

    plt.subplot(1,3,2)
    bound_value = np.max([np.abs(np.nanmin(strf_freq_rate.flatten())),np.abs(np.nanmax(strf_freq_rate.flatten()))])
    if minval == True:    
        im = plt.imshow(strf_freq_rate, aspect=22/128, interpolation=interpolation_,origin='lower',cmap=cmap, vmin=-bound_value, vmax = bound_value)
        fig.colorbar(im, aspect=4, format='%.2f', ticks=[-bound_value,0,bound_value])        
    else:        
        im = plt.imshow(strf_freq_rate, aspect=22/128, interpolation=interpolation_,origin='lower',cmap=cmap, vmin=0, vmax = bound_value)
        fig.colorbar(im, aspect=4, format='%.2f', ticks=[0,bound_value])        
    plt.xticks([])
    plt.yticks([]) 
    plt.axis('off')   
    plt.tight_layout()        

    # plt.xlabel('Rate (Hz)', fontsize=10)
    # plt.ylabel('Frequency (Hz)', fontsize=10)    
    plt.subplot(1,3,3)
    bound_value = np.max([np.abs(np.nanmin(strf_freq_scale.flatten())),np.abs(np.nanmax(strf_freq_scale.flatten()))])    
    if minval == True:    
        im = plt.imshow(strf_freq_scale, aspect=11/128, interpolation=interpolation_,origin='lower',cmap=cmap, vmin=-bound_value, vmax = bound_value)
        fig.colorbar(im, aspect=4, format='%.2f', ticks=[-bound_value,0,bound_value])        
    else:        
        im = plt.imshow(strf_freq_scale, aspect=11/128, interpolation=interpolation_,origin='lower',cmap=cmap, vmin=0, vmax = bound_value)
        fig.colorbar(im, aspect=4, format='%.2f', ticks=[0,bound_value])        

    plt.xticks([])
    plt.yticks([])
    plt.axis('off')   
    plt.tight_layout()        

    # plt.xlabel('Scale (c/o)', fontsize=10)
    # plt.ylabel('Frequency (Hz)', fontsize=10)
    plt.savefig(figname+'.png',bbox_inches='tight')
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig.colorbar(im, cax=cbar_ax)

    if show=='true':
        plt.show()     

def dprimemized(tabA, tabB):
    dprimeTab = (np.mean(np.asarray(tabA),axis=0) - np.mean(np.asarray(tabB),axis=0)) / (np.sqrt(1/2*(np.std(np.asarray(tabA),axis=0)**2+np.std(np.asarray(tabB),axis=0)**2)))
    return dprimeTab

def strf2sumvec(strf,timeDim=False):
 if timeDim:
  strf_scale_rate = np.sum(np.abs(strf), axis=(0, 1))
  strf_freq_rate  = np.sum(np.abs(strf), axis=(0, 2))
  strf_freq_scale = np.sum(np.abs(strf), axis=(0, 3))
 else:
  strf_scale_rate = np.sum(np.abs(strf), axis=0)
  strf_freq_rate  = np.sum(np.abs(strf), axis=1)
  strf_freq_scale = np.sum(np.abs(strf), axis=2)     
 sumvec = np.concatenate((np.ravel(strf_scale_rate), np.ravel(strf_freq_rate), np.ravel(strf_freq_scale)))
 return sumvec

def avgvec2strfavg(avgvec,nbChannels=128,nbRates=22,nbScales=11):
    idxSR = nbRates*nbScales
    idxFR = nbChannels*nbRates
    idxFS = nbChannels*nbScales
    strf_scale_rate = np.reshape(avgvec[:idxSR],(nbScales,nbRates))
    strf_freq_rate = np.reshape(avgvec[idxSR:idxSR+idxFR],(nbChannels,nbRates))
    strf_freq_scale = np.reshape(avgvec[idxSR+idxFR:],(nbChannels,nbScales))
    return strf_scale_rate, strf_freq_rate, strf_freq_scale

def plotStrfavg(strf_scale_rate, strf_freq_rate, strf_freq_scale,aspect_='auto', interpolation_='none',figname='defaut',show='true'):
    plt.suptitle(figname, fontsize=10)
    plt.subplot(1,3,1)
    plt.imshow(strf_scale_rate, aspect=aspect_, interpolation=interpolation_,origin='lower')
    plt.xlabel('Rates (Hz)', fontsize=10)
    plt.ylabel('Scales (c/o)', fontsize=10)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1,3,2)
    plt.imshow(strf_freq_rate, aspect=aspect_, interpolation=interpolation_,origin='lower')
    plt.xticks([])
    plt.yticks([]) 
    plt.xlabel('Rates (Hz)', fontsize=10)
    plt.ylabel('Frequencies (Channels)', fontsize=10)    
    plt.subplot(1,3,3)
    plt.imshow(strf_freq_scale, aspect=aspect_, interpolation=interpolation_,origin='lower')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Scales (c/o)', fontsize=10)
    plt.ylabel('Frequencies (Channels)', fontsize=10)
    plt.savefig(figname+'.png')
    if show=='true':
        plt.show()

def plotStrfavgEqual(strf_scale_rate, strf_freq_rate, strf_freq_scale,aspect_='equal', interpolation_='none',figname='defaut',show='true',title='',cmap='jet'):
    plt.suptitle(title, fontsize=10)
    fig, ax = plt.subplots(nrows=1, ncols=3)
    plt.subplot(1,3,1)
    im = plt.imshow(strf_scale_rate, aspect=22/8, interpolation=interpolation_,origin='lower',cmap=cmap)#, vmin=-1.5, vmax = 0)
    # plt.xlabel('Rate (Hz)', fontsize=10)
    # plt.ylabel('Scale (c/o)', fontsize=10)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')   

    plt.subplot(1,3,2)
    im = plt.imshow(strf_freq_rate, aspect=22/128, interpolation=interpolation_,origin='lower',cmap=cmap)#, vmin=-1.5, vmax = 0)
    plt.xticks([])
    plt.yticks([]) 
    plt.axis('off')   

    # plt.xlabel('Rate (Hz)', fontsize=10)
    # plt.ylabel('Frequency (Hz)', fontsize=10)    
    plt.subplot(1,3,3)
    im = plt.imshow(strf_freq_scale, aspect=8/128, interpolation=interpolation_,origin='lower',cmap=cmap)#, vmin=-1.5, vmax = 0)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')   

    # plt.xlabel('Scale (c/o)', fontsize=10)
    # plt.ylabel('Frequency (Hz)', fontsize=10)
    plt.savefig(figname+'.png',bbox_inches='tight')
    if show=='true':
        plt.show()


def plotStrfavg2(strf_scale_rate, strf_freq_rate, strf_freq_scale,strf_scale_rate2, strf_freq_rate2, strf_freq_scale2,aspect_='equal', interpolation_='none',figname='defaut',show='true',title='',):
    plt.suptitle(title, fontsize=10)
    plt.subplot(2,3,1)
    plt.imshow(strf_scale_rate, aspect=22/8, interpolation=interpolation_,origin='lower')
    # plt.xlabel('Rate (Hz)', fontsize=10)
    # plt.ylabel('Scale (c/o)', fontsize=10)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')   
    plt.subplot(2,3,2)
    plt.imshow(strf_freq_rate, aspect=22/128, interpolation=interpolation_,origin='lower')
    plt.xticks([])
    plt.yticks([]) 
    plt.axis('off')   
    # plt.xlabel('Rate (Hz)', fontsize=10)
    # plt.ylabel('Frequency (Hz)', fontsize=10)    
    plt.subplot(2,3,3)
    plt.imshow(strf_freq_scale, aspect=8/128, interpolation=interpolation_,origin='lower')
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')   
    plt.subplot(2,3,4)
    plt.imshow(strf_scale_rate2, aspect=22/8, interpolation=interpolation_,origin='lower')
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')       
    plt.subplot(2,3,5)
    plt.imshow(strf_freq_rate2, aspect=22/128, interpolation=interpolation_,origin='lower')
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')   
    plt.subplot(2,3,6)
    plt.imshow(strf_freq_scale2, aspect=8/128, interpolation=interpolation_,origin='lower')
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')   
    # plt.xlabel('Scale (c/o)', fontsize=10)
    # plt.ylabel('Frequency (Hz)', fontsize=10)
    plt.savefig(figname+'.png',bbox_inches='tight')
    if show=='true':
        plt.show()
        
def plotStrfDim(strf, dim=0, nElements = 3,label=''):
    nPlots = strf.shape[dim]
    strf = np.swapaxes(strf,0,dim)

    fig, ax = plt.subplots(nrows=int(np.floor(nPlots / nElements)+1), ncols=int(nElements))

    for iPlot in range(nPlots):
        plt.subplot(int(np.floor(nPlots / nElements)+1),nElements,iPlot+1)
        im = plt.imshow(strf[iPlot,:,:], aspect=strf.shape[2]/strf.shape[1], origin='lower',cmap='seismic', vmin=-.5, vmax = .5)
        # plt.show()
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')           
        fig.colorbar(im, aspect=5, format='%.1f')
    plt.subplot(4,3,12)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off') 
    # plt.savefig('./musicspeech_svmrbf_bubbles_method_1a_dim_'+str(label)+'.png',bbox_inches='tight')
    plt.show()      

def PCA_strf_tab(strf_tab_for_pca,strf_tab_to_tranform,nbChannels=128,nbRates=22,nbScales=11,n_components=2):
  strf_pca = []
  strf_2_transform_pca = []
  for iChannel in range(nbChannels):
    reshaped_tab_iChannel = np.reshape(strf_tab_for_pca,(strf_tab_for_pca.shape[0],nbChannels,nbScales*nbRates))[:,iChannel,:]
    reshaped_tab_2_transform_iChannel = np.reshape(strf_tab_to_tranform,(strf_tab_to_tranform.shape[0],nbChannels,nbScales*nbRates))[:,iChannel,:]
    pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(reshaped_tab_iChannel)
    strf_pca.append(pca.transform(reshaped_tab_iChannel))
    strf_2_transform_pca.append(pca.transform(reshaped_tab_2_transform_iChannel))
  strf_pca_transformed = np.reshape(np.array(strf_pca),(strf_tab_for_pca.shape[0],nbChannels*n_components))
  strf_2_transformed_pca_transformed = np.reshape(np.array(strf_2_transform_pca),(strf_tab_to_tranform.shape[0],nbChannels*n_components))
  return strf_pca_transformed, strf_2_transformed_pca_transformed