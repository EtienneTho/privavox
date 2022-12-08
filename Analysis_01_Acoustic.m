clear all
clearvars ;
clc;

% initialize sound path
addpath(genpath('./')); 
fileListWithin  = dir(strcat('./out/out_01_Acoustic_Analysis/*_sub.mat')) ;
fileListBetween = dir(strcat('./out/out_01_Acoustic_Analysis/AcousticAnalysis_BetweenSubjects.mat')) ;

fid=fopen('./out/ScaleRate/results_Acoustic.txt','w');

% strf parameters
frequencies = 440 * 2 .^ ((-31:96)/24) ;
rates = [-32, -22.6, -16, -11.3, -8, -5.70, -4, -2, -1, -.5, -.25, .25, .5, 1, 2, 4, 5.70, 8, 11.3, 16, 22.6, 32] ;
scales = [0.71, 1.0, 1.41, 2.00, 2.83, 4.00, 5.66, 8.00] ;

% stanford sleepiness reports
SSS = [[1,3,3,1,1,1,1,3,1,2,1,3,4,3,2,3,1,1,3,1,1,1];...
             [1,2,1,2,1,1,1,1,1,2,1,3,4,3,2,1,1,1,3,1,1,1];...
             [1,1,3,2,1,2,1,1,1,1,1,1,3,3,2,4,2,3,4,1,2,2];...
             [1,2,1,3,1,3,1,1,1,1,1,3,4,5,2,2,2,3,3,1,3,2];...
             [5,3,5,1,3,5,2,5,2,2,2,2,2,3,2,5,3,3,3,3,3,3];...
             [5,3,3,1,2,6,3,3,2,2,2,2,2,3,2,3,2,2,3,3,2,2];...
             [3,3,6,2,2,6,4,3,6,1,3,3,6,3,2,4,4,4,4,2,4,2];...
             [2,2,6,1,3,5,4,2,3,3,2,3,6,6,3,5,3,5,4,2,6,3]]'  ;  
sleepLoss = mean(SSS(:,(5:8)),2) ;     

vecSubject = (1:22); 

% initialisations
tabMasks = zeros(length(vecSubject),128*8*22) ;
tabSubject = zeros(1,length(vecSubject)) ;
averagedCorr = zeros(1,length(vecSubject)) ;
maskTot = [] ;
stanfordAllMaps = [] ;
N_seed = 1 ;

% full
tabMasks_sub = squeeze(reshape(tabMasks,length(vecSubject),22,8,128)) ;
Ny = 8;
Nx = 22;
Nz = 128;

tabMasks = reshape(tabMasks_sub,length(vecSubject),Nx*Ny*Nz) ;
dim2avg = 2 ;

%% freq-rate
X = rates;
Y = frequencies;
freqLabels = fliplr([250 500 1000 2000 4000]) ;
freqPos    = sort(128-floor(24*log2(freqLabels/440)+36))*2^4 ;
dim2avg = 2 ;
%% between subjects
load('./out/out_01_Acoustic_Analysis/AcousticAnalysis_BetweenSubjects.mat')
    toPlot_mp = rot90(squeeze(mean(reshape(mean_post,Nx,Ny,Nz),dim2avg))) ;
    toPlot = interp2(toPlot_mp,4);
    XTick_pos    = linspace(1,numrc(toPlot,2),6) ;
    XTick_labels = (X(round(linspace(1,length(X),6)))) ;
%     YTick_pos    = linspace(1,numrc(toPlot,1),3) ;
%     YTick_labels = (fliplr(Y(round(linspace(1,length(Y),3))))) ;
    YTick_pos    = freqPos ;
    YTick_labels = freqLabels ;
    plot_sr(toPlot,500*[-1e-3 1e-3], 'Rate (Hz)', 'Frequency (Hz)',... 
                       XTick_pos, XTick_labels,...
                       YTick_pos, YTick_labels, 0) ;
    colormap(customcolormap_preset('red-white-blue'));

    saveas(gcf,['./out/Acoustics/mean_post/freqRate/meanPost_betweenSubjects'],'epsc')
    toPlot_pr = rot90(squeeze(mean(reshape(mean_pre,Nx,Ny,Nz),dim2avg))) ;
    toPlot = interp2(toPlot_pr,4);
    XTick_pos    = linspace(1,numrc(toPlot,2),6) ;
    XTick_labels = X(round(linspace(1,length(X),6))) ;
    YTick_pos    = freqPos ;
    YTick_labels = freqLabels ;
    plot_sr(toPlot,500*[-1e-3 1e-3], 'Rate (Hz)', 'Frequency (Hz)',... 
                       XTick_pos, XTick_labels,...
                       YTick_pos, YTick_labels, 0) ;
    colormap(customcolormap_preset('red-white-blue'));
    saveas(gcf,['./out/Acoustics/mean_pre/freqRate/meanPre_betweenSubjects'],'epsc')
    
%     toPlot = rot90(squeeze(mean(reshape(diff_mean,Nx,Ny,Nz),dim2avg))) ;
    toPlot = reshape(2 * abs(toPlot_mp-toPlot_pr) ./ (toPlot_mp+toPlot_pr),Nz,Nx) ;
    max(toPlot(:))
    toPlot = interp2(toPlot,4);    
    XTick_pos    = linspace(1,numrc(toPlot,2),6) ;
    XTick_labels = X(round(linspace(1,length(X),6))) ;
    YTick_pos    = freqPos ;
    YTick_labels = freqLabels ;
    plot_sr(toPlot,120*[-1e-3 1e-3], 'Rate (Hz)', 'Frequency (Hz)',... 
                       XTick_pos, XTick_labels,...
                       YTick_pos, YTick_labels, 0) ;
    colormap(customcolormap_preset('red-white-blue'));
    saveas(gcf,['./out/Acoustics/diff_mean/freqRate/diffMean_betweenSubjects'],'epsc')
    
%% within subjects    
%% mean post freq-rate
tabMaskAcoustic = [] ;
maxFR = [] ;

for iFile = 1:length(vecSubject) 
    
    load(fileListWithin(iFile).name) ;
    % mean_post
    toPlot_mp = rot90(squeeze(mean(reshape(mean_post,Nx,Ny,Nz),dim2avg))) ;
    tabMaskAcoustic_meanPost = [tabMaskAcoustic; mean(toPlot,2)'] ;
    toPlot = interp2(toPlot_mp,4) ;
    XTick_pos    = linspace(1,numrc(toPlot,2),6) ;
    XTick_labels = X(round(linspace(1,length(X),6))) ;
    YTick_pos    = freqPos ;
    YTick_labels = freqLabels ;
    plot_sr(toPlot,500*[-1e-3 1e-3], 'Rate (Hz)', 'Frequency (Hz)',... 
                       XTick_pos, XTick_labels,...
                       YTick_pos, YTick_labels, 1) ;
    colormap(customcolormap_preset('red-white-blue'));

    title(num2str(iSubject));
    saveas(gcf,['./out/Acoustics/mean_post/freqRate/mean_post_',num2str(iFile),'_sub#_',num2str(iSubject)],'epsc')
    
    % mean_pre    
    toPlot_pr = rot90(squeeze(mean(reshape(mean_pre,Nx,Ny,Nz),dim2avg))) ;
    tabMaskAcoustic_meanPre = [tabMaskAcoustic; mean(toPlot,2)'] ;
    toPlot = interp2(toPlot_pr,4) ;    
    XTick_pos    = linspace(1,numrc(toPlot,2),6) ;
    XTick_labels = X(round(linspace(1,length(X),6))) ;
%     YTick_pos    = linspace(1,numrc(toPlot,1),7) ;
%     YTick_labels = fliplr(Y(round(linspace(1,length(Y),7)))) ;
    YTick_pos    = freqPos ;
    YTick_labels = freqLabels ;
    plot_sr(toPlot,500*[-1e-3 1e-3], 'Rate (Hz)', 'Frequency (Hz)',... 
                       XTick_pos, XTick_labels,...
                       YTick_pos, YTick_labels, 1) ;
    colormap(customcolormap_preset('red-white-blue'));

    title(num2str(iSubject));
    saveas(gcf,['./out/Acoustics/mean_pre/freqRate/mean_pre_',num2str(iFile),'_sub#_',num2str(iSubject)],'epsc')
    
    % diff_mean
    toPlot = reshape(2 * abs(toPlot_mp-toPlot_pr) ./ (toPlot_mp+toPlot_pr),Nz,Nx) ;    
    maxFR = [maxFR max(toPlot(:))] ;
    tabMaskAcoustic_diffMean = [tabMaskAcoustic; mean(toPlot,2)'] ;
    toPlot_ = interp2(toPlot,4) ;    
    XTick_pos    = linspace(1,numrc(toPlot_,2),6) ;
    XTick_labels = X(round(linspace(1,length(X),6))) ;
%     YTick_pos    = linspace(1,numrc(toPlot_,1),7) ;
%     YTick_labels = fliplr(Y(round(linspace(1,length(Y),7)))) ;
    YTick_pos    = freqPos ;
    YTick_labels = freqLabels ;
    plot_sr(toPlot_,500*[-1e-3 1e-3], 'Rate (Hz)', 'Frequency (Hz)',... 
                       XTick_pos, XTick_labels,...
                       YTick_pos, YTick_labels, 1) ;
    colormap(customcolormap_preset('red-white-blue'));

    title(num2str(iSubject));
    tabMaskAcoustic_meanPost = [tabMaskAcoustic; mean(toPlot,2)'] ;
    saveas(gcf,['./out/Acoustics/diff_mean/freqRate/diff_mean',num2str(iFile),'_sub#_',num2str(iSubject)],'epsc')
    close all    
    % diff_mean
end

%% scale-rate
X = rates;
Y = scales;
scaleLabels = fliplr([1 2 4 8]) ;
scalePos    = sort(9-[2 4 6 8])*(2^4-1) ;
dim2avg     = 3 ;
%% between subjects
    load('./out/out_01_Acoustic_Analysis/AcousticAnalysis_BetweenSubjects.mat')
    toPlot_mp = rot90(squeeze(mean(reshape(mean_post,Nx,Ny,Nz),dim2avg))) ;
    toPlot = interp2(toPlot_mp,4);
    XTick_pos    = linspace(1,numrc(toPlot,2),6) ;
    XTick_labels = X(round(linspace(1,length(X),6))) ;
    % YTick_pos    = linspace(1,numrc(toPlot,1),7) ;
    % YTick_labels = fliplr(Y(round(linspace(1,length(Y),7)))) ;
    YTick_pos    = scalePos ;
    YTick_labels = scaleLabels ;
    plot_sr(toPlot,500*[-1e-3 1e-3], 'Rate (Hz)', 'Scale (cyc/oct)',... 
                       XTick_pos, XTick_labels,...
                       YTick_pos, YTick_labels, 0) ;
    colormap(customcolormap_preset('red-white-blue'));
    saveas(gcf,['./out/Acoustics/mean_post/scaleRate/meanPost_betweenSubjects'],'epsc')
    
    toPlot_pr = rot90(squeeze(mean(reshape(mean_pre,Nx,Ny,Nz),dim2avg))) ;
    toPlot = interp2(toPlot_pr,4);
    XTick_pos    = linspace(1,numrc(toPlot,2),6) ;
    XTick_labels = X(round(linspace(1,length(X),6))) ;
    %YTick_pos    = linspace(1,numrc(toPlot,1),8) ;
    %YTick_labels = fliplr(Y(round(linspace(1,length(Y),8)))) ;
    YTick_pos    = scalePos ;
    YTick_labels = scaleLabels ;    
    plot_sr(toPlot,500*[-1e-3 1e-3], 'Rate (Hz)', 'Scale (cyc/oct)',... 
                       XTick_pos, XTick_labels,...
                       YTick_pos, YTick_labels, 0) ;
    colormap(customcolormap_preset('red-white-blue'));    
    saveas(gcf,['./out/Acoustics/mean_pre/scaleRate/meanPre_betweenSubjects'],'epsc')
    
    toPlot_diff = reshape(2 * abs(toPlot_mp-toPlot_pr) ./ (toPlot_mp+toPlot_pr),Ny,Nx) ;
    max(toPlot_diff(:))
    toPlot = interp2(toPlot_diff,4) ;
    XTick_pos    = linspace(1,numrc(toPlot,2),6) ;
    XTick_labels = X(round(linspace(1,length(X),6))) ;
    %YTick_pos    = linspace(1,numrc(toPlot,1),7) ;
    %YTick_labels = fliplr(Y(round(linspace(1,length(Y),7)))) ;
    YTick_pos    = scalePos ;
    YTick_labels = scaleLabels ;
    plot_sr(toPlot,50*[-1e-3 1e-3], 'Rate (Hz)', 'Scale (cyc/oct)',... 
                       XTick_pos, XTick_labels,...
                       YTick_pos, YTick_labels, 0) ;
    colormap(customcolormap_preset('red-white-blue'));
    saveas(gcf,['./out/Acoustics/diff_mean/scaleRate/diffMean_betweenSubjects'],'epsc')
    

%% within subjects    
%% mean post scale-rate
tabMaskAcoustic_diffMean = [] ;
tabMaskAcoustic_meanPost = [] ;
tabMaskAcoustic_meanPre  = [] ;
maxSR = [] ;

for iFile = 1:length(vecSubject) 
    iFile
    load(fileListWithin(iFile).name) ;
    % mean_post
    toPlot_mp = rot90(squeeze(mean(reshape(mean_post,Nx,Ny,Nz),dim2avg))) ;
    tabMaskAcoustic_meanPost = [tabMaskAcoustic_meanPost; mean(toPlot_mp,2)'] ;
    toPlot = interp2(toPlot_mp,4) ;
    XTick_pos    = linspace(1,numrc(toPlot,2),16) ;
    XTick_labels = X(round(linspace(1,length(X),16))) ;
%     YTick_pos    = linspace(1,numrc(toPlot,1),7) ;
%     YTick_labels = fliplr(Y(round(linspace(1,length(Y),7)))) ;
    YTick_pos    = scalePos ;
    YTick_labels = scaleLabels ;
    plot_sr(toPlot,500*[-1e-3 1e-3], 'Rate (Hz)', 'Scale (cyc/oct)',... 
                       XTick_pos, XTick_labels,...
                       YTick_pos, YTick_labels, 1) ;
    colormap(customcolormap_preset('red-white-blue'));

    title(num2str(iSubject));
    saveas(gcf,['./out/Acoustics/mean_post/scaleRate/mean_post_',num2str(iFile),'_sub#_',num2str(iSubject)],'epsc')
    
    % mean_pre    
    toPlot_pr = rot90(squeeze(mean(reshape(mean_pre,Nx,Ny,Nz),dim2avg))) ;
    tabMaskAcoustic_meanPre = [tabMaskAcoustic_meanPre; mean(toPlot_pr,2)'] ;
    toPlot = interp2(toPlot_pr,4) ;    
    XTick_pos    = linspace(1,numrc(toPlot,2),16) ;
    XTick_labels = X(round(linspace(1,length(X),16))) ;
%     YTick_pos    = linspace(1,numrc(toPlot,1),7) ;
%     YTick_labels = fliplr(Y(round(linspace(1,length(Y),7)))) ;
    YTick_pos    = scalePos ;
    YTick_labels = scaleLabels ;
    plot_sr(toPlot,500*[-1e-3 1e-3], 'Rate (Hz)', 'Scale (cyc/oct)',... 
                       XTick_pos, XTick_labels,...
                       YTick_pos, YTick_labels, 1) ;
    colormap(customcolormap_preset('red-white-blue'));

    title(num2str(iSubject));
    saveas(gcf,['./out/Acoustics/mean_pre/scaleRate/mean_pre_',num2str(iFile),'_sub#_',num2str(iSubject)],'epsc')

    % diff_mean
    toPlot = reshape(2 * abs(toPlot_mp-toPlot_pr) ./ (toPlot_mp+toPlot_pr),Ny,Nx) ;    
    
    tabMaskAcoustic_diffMean = [tabMaskAcoustic_diffMean; toPlot(:)'] ;
    maxSR = [maxSR max(tabMaskAcoustic_diffMean(:))] ;
    
    %toPlot = interp2(toPlot,4) ;    
    XTick_pos    = linspace(1,numrc(toPlot,2),16) ;
    XTick_labels = X(round(linspace(1,length(X),16))) ;
%     YTick_pos    = linspace(1,numrc(toPlot,1),7) ;
%     YTick_labels = fliplr(Y(round(linspace(1,length(Y),7)))) ;
    YTick_pos    = scalePos ;
    YTick_labels = scaleLabels ;
    plot_sr(toPlot,300*[-1e-3 1e-3], 'Rate (Hz)', 'Scale (cyc/oct)',... 
                       XTick_pos, XTick_labels,...
                       YTick_pos, YTick_labels, 1) ;
    colormap(customcolormap_preset('red-white-blue'));

    title(num2str(iSubject));
    saveas(gcf,['./out/Acoustics/diff_mean/scaleRate/diff_mean',num2str(iFile),'_sub#_',num2str(iSubject)],'epsc')
    close all
    % diff_mean
end



%% continuum subjects BAcc
figure;
%[~, i_PCA1] = sort(scores(:,1)) ;
path = './out/benchmark_pca/subject_level/30_PCs/';
load(strcat(path,'/BAcc_3D.mat'));
mean_acc = mean(tabBAcc_3d,2) ;
[~, index] = sort(mean_acc) ;

for iFile = 1:length(vecSubject) 
    iFile
    subplot(3,8,iFile)    
    toPlot = reshape(tabMaskAcoustic_diffMean(index(iFile),:),Ny,Nx) ;
%     IPCA_PL = inversePcaPythonLike([scores(i_PCA1(iFile),1) scores(iFile,2)], eigenVectors, 0, mu, sigma, mean_) ;
%     canonicalMap_reconstruct = rot90(mean(reshape(IPCA_PL(1,:),Nx,Ny,Nz),dim2avg)) ;
%     subplot(121)
    toPlot = interp2(toPlot,4) ;
    imagesc(toPlot,[-.2 .2]);
    if iFile ~= 1
        axis('square')
        set(gcf,'position',[10,10,450,230])
%     subplot(122)
%     imagesc(canonicalMap_reconstruct,[-.01 .01]) ;
        title(num2str((iFile))) ;
        set(gca,'xtick',[],'ytick',[]) 
        colormap(customcolormap_preset('red-white-blue'));
    else
        title(num2str((iFile))) ;
        colormap(customcolormap_preset('red-white-blue'));        
        set(gcf,'position',[10,10,450,230])
        axis('square')
        set(gca,'xtick',[1 337],'XTickLabel', [-32 32],'ytick',[1 113],'YTickLabel', [8 1],'fontsize',8)        
        xlabel('Rate (Hz)')
        ylabel('Scale (c/o)')
    end
end
colorbar('Position',[0.91 0.168 0.013 0.7]); 

saveas(gcf,['./out/Acoustics/diff_mean/scaleRate/diffMeanSubjectMasks_sorted_from_BAcc'],'epsc')

%% Functions
%% pca functions

function [eigenVectors, mu, sigma, mean_, scores,explained] = pcaPythonLike(X, ncomp, whitened)
    [eigenVectors, scores, ~,~,explained,mean_] = pca(X);
    mu = 0 ;
    sigma = 0 ;
    if whitened == 0
        [~,mu,sigma] = zscore(scores) ;
    end 
    eigenVectors = eigenVectors(:,1:ncomp) ;
    mu = mu(1:ncomp) ;
    sigma = sigma(1:ncomp) ;
end

function [output] = inversePcaPythonLike(X, eigenVectors, whitened, mu, sigma, mean_)
    if whitened == 0
        X = X .* sigma + mu ;
    end
    output = X * eigenVectors' + repmat(mean_,length(X),1) ;
end


%% function that gives the coefficient of determination
function R2 = r2_score(X,Y)
% Filname: 'rsquare.m'. This file can be used for directly calculating
% the coefficient of determination (R2) of a dataset.
%
% Two input arguments: 'X' and 'Y'
% One output argument: 'R2'
%
% X:    Vector of x-parameter
% Y:    Vector of y-paramter
% R2:   Coefficient of determination
%
% Input syntax: rsquare(X,Y)
%
% Developed by Joris Meurs BASc (2016)

% Limitations
if length(X) ~= length(Y), error('Vector should be of same length');end
if nargin < 2, error('Not enough input parameters');end
if nargin > 2, error('Too many input parameters');end

% Linear regression according to the model: a + bx
A = [ones(length(X),1) X];
b = pinv(A'*A)*A'*Y;
Y_hat = A*b;

% Calculation of R2 according to the formula: SSreg/SStot
SSreg = sum((Y_hat - mean(Y)).^2);
SStot = sum((Y - mean(Y)).^2);
R2 = SSreg/SStot;

% Output limitations
if R2 > 1, error('Irregular value, check your data');end
if R2 < 0, error('Irregular value, check your data');end
end

%% function that give the number of rows and columns of an array with dim=1 for row
% and dim=2 for columns
function nb = numrc(array,dim)
[r,c]=size(array) ;
nb = r;
    if dim == 1
        nb = r;
    else
        nb = c;
    end
end
