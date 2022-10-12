clear all
clearvars ;
clc;
% initialize sound path
addpath(genpath('./')); 
fileList = dir(strcat('./out/out_03_classicationSubjectLevel_AllMaps/*_3d.mat')) ;
fid=fopen('./out/FreqRate/results_freqRate.txt','w');

% strf parameters
frequencies = 440 * 2 .^ ((-31:96)/24) ;
rates = [-32, -22.6, -16, -11.3, -8, -5.70, -4, -2, -1, -.5, -.25, .25, .5, 1, 2, 4, 5.70, 8, 11.3, 16, 22.6, 32] ;
scales = [0.71, 1.0, 1.41, 2.00, 2.83, 4.00, 5.66, 8.00] ;

load('BAcc_3D.mat'); % load balanced accuracies
vecSubject = (1:22); 

% initialisations
tabMasks = zeros(length(vecSubject),128*8*22) ;
tabSubject = zeros(1,length(vecSubject)) ;
averagedCorr = zeros(1,length(vecSubject)) ;
maskTot = [] ;
stanfordAllMaps = [] ;
N_seed = 1 ;

% load canonical maps
for iFile = 1:length(vecSubject) %1:length(fileList) 
    load(fileList(vecSubject(iFile)).name);
    canonicalMap = nanmean(canonicalAllMaps(:,:),1) ;
    tabMasks(iFile,:) = canonicalMap(:) ;
    tabSubject(iFile) = iSubject ;
end

%% def sub representations Scale-Rate / Frequency-Rate / Frequency-Scale

% full
tabMasks_sub = squeeze(reshape(tabMasks,length(vecSubject),22,8,128)) ;
Ny = 8;
Nx = 22;
Nz = 128;

tabMasks = reshape(tabMasks_sub,length(vecSubject),Nx*Ny*Nz) ;
dim2avg = 2 ;
X = rates;
Y = frequencies;

%% Interpretations
tabInterpretationsAll = [] ;
tabInterpretationsSR = [] ;
tabInterpretationsFR = [] ;
tabInterpretationsFS = [] ;
tabInterpretationsF  = [] ;
tabInterpretationsS  = [] ;
tabInterpretationsR  = [] ;

for iFile = 1:length(vecSubject) 
    load(fileList(iFile).name);
    toPlotSR = rot90(squeeze(mean(reshape(tabMasks(iFile,:),Nx,Ny,Nz),3))) ;
    toPlotFR = rot90(squeeze(mean(reshape(tabMasks(iFile,:),Nx,Ny,Nz),2))) ;
    toPlotFS = rot90(squeeze(mean(reshape(tabMasks(iFile,:),Nx,Ny,Nz),1))) ;
    toPlotF  = mean(rot90(squeeze(mean(reshape(tabMasks(iFile,:),Nx,Ny,Nz),1))),2) ;
    toPlotS  = mean(rot90(squeeze(mean(reshape(tabMasks(iFile,:),Nx,Ny,Nz),1))),1) ;
    toPlotR  = mean(rot90(squeeze(mean(reshape(tabMasks(iFile,:),Nx,Ny,Nz),3))),1) ;
    
    tabInterpretationsAll = [tabInterpretationsAll; tabMasks(iFile,:)] ;
    tabInterpretationsSR = [tabInterpretationsSR; toPlotSR(:)'] ;
    tabInterpretationsFR = [tabInterpretationsFR; toPlotFR(:)'] ;
    tabInterpretationsFS = [tabInterpretationsFS; toPlotFS(:)'] ;
    tabInterpretationsF  = [tabInterpretationsF; toPlotF(:)'] ;
    tabInterpretationsS  = [tabInterpretationsS; toPlotS(:)'] ;
    tabInterpretationsR  = [tabInterpretationsR; toPlotR(:)'] ;
    
end

%% Acoustic
fileListWithin  = dir(strcat('./out/out_01_Acoustic_Analysis/*_sub.mat')) ;
tabMaskAcoustic_diffMeanAll = [] ;
tabMaskAcoustic_diffMeanSR = [] ;
tabMaskAcoustic_diffMeanFR = [] ;
tabMaskAcoustic_diffMeanFS = [] ;
tabMaskAcoustic_diffMeanF  = [] ;
tabMaskAcoustic_diffMeanS  = [] ;
tabMaskAcoustic_diffMeanR  = [] ;

for iFile = 1:length(vecSubject) 
    iFile
    load(fileListWithin(iFile).name) ;
    
    % whole tensor
    % mean_post
    toPlot_mp = mean_post ;
    % mean_pre    
    toPlot_pr = mean_pre ;
    % diff_mean
    toPlot = 2 * (toPlot_mp-toPlot_pr) ./ (toPlot_mp+toPlot_pr) ;    
    tabMaskAcoustic_diffMeanAll = [tabMaskAcoustic_diffMeanAll; toPlot] ;
    
    % scale rate
    % mean_post
    toPlot_mp = rot90(squeeze(mean(reshape(mean_post,Nx,Ny,Nz),3))) ;
    % mean_pre    
    toPlot_pr = rot90(squeeze(mean(reshape(mean_pre,Nx,Ny,Nz),3))) ;
    % diff_mean
    toPlot = 2 * (toPlot_mp-toPlot_pr) ./ (toPlot_mp+toPlot_pr) ;    
    tabMaskAcoustic_diffMeanSR = [tabMaskAcoustic_diffMeanSR; toPlot(:)'] ;

    % freq rate
    % mean_post
    toPlot_mp = rot90(squeeze(mean(reshape(mean_post,Nx,Ny,Nz),2))) ;
    % mean_pre    
    toPlot_pr = rot90(squeeze(mean(reshape(mean_pre,Nx,Ny,Nz),2))) ;
    % diff_mean
    toPlot = 2 * (toPlot_mp-toPlot_pr) ./ (toPlot_mp+toPlot_pr) ;    
    tabMaskAcoustic_diffMeanFR = [tabMaskAcoustic_diffMeanFR; toPlot(:)'] ;    

    % freq scale
    % mean_post
    toPlot_mp = rot90(squeeze(mean(reshape(mean_post,Nx,Ny,Nz),1))) ;
    % mean_pre    
    toPlot_pr = rot90(squeeze(mean(reshape(mean_pre,Nx,Ny,Nz),1))) ;
    % diff_mean
    toPlot = 2 * (toPlot_mp-toPlot_pr) ./ (toPlot_mp+toPlot_pr) ;    
    tabMaskAcoustic_diffMeanFS = [tabMaskAcoustic_diffMeanFS; toPlot(:)'] ;       

    % freq 
    % mean_post
    toPlot_mp = mean(rot90(squeeze(mean(reshape(mean_post,Nx,Ny,Nz),1))),2) ;
    % mean_pre    
    toPlot_pr = mean(rot90(squeeze(mean(reshape(mean_pre,Nx,Ny,Nz),1))),2) ;
    % diff_mean
    toPlot = 2 * (toPlot_mp-toPlot_pr) ./ (toPlot_mp+toPlot_pr) ;    
    tabMaskAcoustic_diffMeanF = [tabMaskAcoustic_diffMeanF; toPlot(:)'] ; 
    
    %  scale
    % mean_post
    toPlot_mp = mean(rot90(squeeze(mean(reshape(mean_post,Nx,Ny,Nz),1))),1) ;
    % mean_pre    
    toPlot_pr = mean(rot90(squeeze(mean(reshape(mean_pre,Nx,Ny,Nz),1))),1) ;
    % diff_mean
    toPlot = 2 * (toPlot_mp-toPlot_pr) ./ (toPlot_mp+toPlot_pr) ;    
    tabMaskAcoustic_diffMeanS = [tabMaskAcoustic_diffMeanS; toPlot(:)'] ;  

    % rate
    % mean_post
    toPlot_mp = mean(rot90(squeeze(mean(reshape(mean_post,Nx,Ny,Nz),3))),1) ;
    % mean_pre    
    toPlot_pr = mean(rot90(squeeze(mean(reshape(mean_pre,Nx,Ny,Nz),3))),1) ;
    % diff_mean
    toPlot = 2 * (toPlot_mp-toPlot_pr) ./ (toPlot_mp+toPlot_pr) ;    
    tabMaskAcoustic_diffMeanR = [tabMaskAcoustic_diffMeanR; toPlot(:)'] ;

end

%% raw correlation
tabCorrelation_Interp_AcousticAl = [] ;
tabCorrelation_Interp_AcousticSR = [] ;
tabCorrelation_Interp_AcousticFR = [] ;
tabCorrelation_Interp_AcousticFS = [] ;
tabCorrelation_Interp_AcousticF  = [] ;
tabCorrelation_Interp_AcousticS  = [] ;
tabCorrelation_Interp_AcousticR  = [] ;


for iFile = 1:length(vecSubject) 
    tabCorrelation_Interp_AcousticAll(iFile) = corr(tabMaskAcoustic_diffMeanAll(iFile,:)', tabInterpretationsAll(iFile,:)') ;
    tabCorrelation_Interp_AcousticSR(iFile) = corr(tabMaskAcoustic_diffMeanSR(iFile,:)',   tabInterpretationsSR(iFile,:)') ;
    tabCorrelation_Interp_AcousticFR(iFile) = corr(tabMaskAcoustic_diffMeanFR(iFile,:)',   tabInterpretationsFR(iFile,:)') ;
    tabCorrelation_Interp_AcousticFS(iFile) = corr(tabMaskAcoustic_diffMeanFS(iFile,:)',   tabInterpretationsFS(iFile,:)') ;
    tabCorrelation_Interp_AcousticF(iFile) = corr(tabMaskAcoustic_diffMeanF(iFile,:)',   tabInterpretationsF(iFile,:)') ;
    tabCorrelation_Interp_AcousticS(iFile) = corr(tabMaskAcoustic_diffMeanS(iFile,:)',   tabInterpretationsS(iFile,:)') ;
    tabCorrelation_Interp_AcousticR(iFile) = corr(tabMaskAcoustic_diffMeanR(iFile,:)',   tabInterpretationsR(iFile,:)') ;
end

figure;
toPlot_all = tabCorrelation_Interp_AcousticAll ;
toPlot_sr  = tabCorrelation_Interp_AcousticSR ;
toPlot_fr  = tabCorrelation_Interp_AcousticFR ;
toPlot_fs  = tabCorrelation_Interp_AcousticFS ;
toPlot_f   = tabCorrelation_Interp_AcousticF ;
toPlot_s   = tabCorrelation_Interp_AcousticS ;
toPlot_r   = tabCorrelation_Interp_AcousticR ;

plot(toPlot_all.^2);hold on;
plot(toPlot_sr.^2);hold on;
plot(toPlot_fr.^2);hold on;
plot(toPlot_fs.^2);hold on;
plot(toPlot_f.^2);hold on;
plot(toPlot_s.^2);hold on;
plot(toPlot_r.^2);hold on;
xlabel('Subject #')    
ylabel('r^2 (Pearson)')
grid on;
grid(gca,'minor')
legend('Full','Scale-Rate','Frequency-Rate','Frequency-Scale','Frequency','Scale','Rate')
axis([0 length(vecSubject)+1 0 1 ])
axis('square')
saveas(gcf,['./out/ComparisonAcousticInterpretations/rawCorrelationBetweenMaps'],'epsc')
close all;


tab_correlation_interpretation_acoustic = ...
    [toPlot_all;
    toPlot_sr
    toPlot_fr
    toPlot_fs
    toPlot_f
    toPlot_r
    toPlot_s];

