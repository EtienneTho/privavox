clear all
clearvars ;
clc;
% initialize sound path
addpath(genpath('./')); 
%fileList = dir(strcat('./out/out_03_classicationSubjectLevel_AllMaps/*_3d.mat')) ;
pc_nb = [3 5 10 15 20 25 30 35 40 45 50 55 60 65 70] ;
tabMasksAll = zeros(length(pc_nb),128*8*22,22) ;
tabBAccAll = zeros(length(pc_nb),22) ;

for iPC = 1:length(pc_nb)
    iPC
    path = ['./out/benchmark_pca/', num2str(pc_nb(iPC)) ,'_PCs/'] ;
    fileList = dir(strcat(path,'/*_3d.mat')) ;
    fid=fopen('./out/results_SSS_PCA.txt','w');

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

    load(strcat(path,'/BAcc_3D.mat')); % load balanced accuracies
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
        load(strcat(path,fileList(vecSubject(iFile)).name));
        canonicalMap = nanmean(canonicalAllMaps(:,:),1) ;
        [rr,cc] = size(canonicalAllMaps);
        tabMasks(iFile,:) = canonicalMap(:) ;
        tabSubject(iFile) = iSubject ;
        stanfordAllMaps = [stanfordAllMaps; repmat(SSS(iFile,:),[N_seed 1])] ;
        triu_ = triu(corr(canonicalAllMaps(end-(rr-1):end,:)'),1) ;
        triu_(triu_==0) = [] ;
        averagedCorr(iFile) = nanmean(triu_) ;
        tabMasksAll(iPC,:,iFile) = canonicalMap(:) ;
    end
    load(strcat(path,'/BAcc_3D.mat'));        
    tabBAccAll(iPC,:) = mean(tabBAcc_3d,2) ;

end    

%% stability
% stability interpretation
figure
subplot(121)
tabMeanCorr = zeros(15,22) ;
for iSub = 1:22
    subMat = squeeze(tabMasksAll(:,:,iSub)) ;
    pDistMat = pdist(subMat,'correlation') ;
%     imagesc(1-squareform(pDistMat))
    tabMeanCorr(:,iSub) = mean(1-squareform(pDistMat)) ;
end

errorbar(pc_nb,mean(tabMeanCorr,2),std(tabMeanCorr,[],2),'linewidth',2) ;
xlabel('Number of PCs')
ylabel('Average pairwise correlation')
grid on;
axis square
axis([1 75 0.5 1]);
set(gca, 'fontsize',18); % 20 ticks

% stability BAcc
subplot(122)
errorbar(pc_nb,mean(tabBAccAll,2),std(tabBAccAll,[],2),'linewidth',2) ;
xlabel('Number of PCs')
ylabel('Average BAcc')
grid on;
axis square
axis([1 75 0.5 1]);
set(gca, 'fontsize',18); % 20 ticks
saveas(gcf,['./out/benchmarkPCABAcc'],'epsc')

