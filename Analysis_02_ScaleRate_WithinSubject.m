clear all
clearvars ;
clc;
% initialize sound path
addpath(genpath('./')); 
%fileList = dir(strcat('./out/out_03_classicationSubjectLevel_AllMaps/*_3d.mat')) ;
path = './out/benchmark_pca/subject_level/30_PCs/';
%%
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
warning ('off','all');
% initialisations
tabMasks = zeros(length(vecSubject),128*8*22) ;
tabSubject = zeros(1,length(vecSubject)) ;
averagedCorr   = zeros(1,length(vecSubject)) ;
averagedCorr_2 = zeros(1,length(vecSubject)) ;
averaged_p     = zeros(1,length(vecSubject)) ;
averaged_bf10  = zeros(1,length(vecSubject)) ;
maskTot = [] ;
stanfordAllMaps = [] ;
N_seed = 1 ;

% load canonical maps
for iFile = 1:length(vecSubject) %1:length(fileList) 
    iFile
    load(strcat(path,fileList(vecSubject(iFile)).name));
    canonicalMap = nanmean(canonicalAllMaps(:,:),1) ;
    [rr,cc] = size(canonicalAllMaps);
    tabMasks(iFile,:) = canonicalMap(:) ;
    tabSubject(iFile) = iSubject ;
    stanfordAllMaps = [stanfordAllMaps; repmat(SSS(iFile,:),[N_seed 1])] ;
    [r_tab,p_tab,BF10_tab] = corrBF10_tab(canonicalAllMaps(end-(rr-1):end,:)) ;
    
    triu_ = triu(corr(canonicalAllMaps(end-(rr-1):end,:)'),1) ;
    triu_(triu_==0) = [] ;
    averagedCorr(iFile) = nanmean(triu_) ;
    
    triu_ = triu(r_tab,1) ;
    triu_(triu_==0) = [] ;    
    averagedCorr_2(iFile) = nanmean(triu_) ;
    
    triu_ = triu(p_tab,1) ;
    triu_(triu_==0) = [] ;    
    averaged_p(iFile) = nanmean(triu_) ;    
    
    triu_ = triu(BF10_tab,1) ;
    triu_(triu_==0) = [] ;    
    averaged_bf10(iFile) = nanmean(triu_) ;    
end


%% PCA on Masks
figure
load(strcat(path,'/BAcc_3D.mat'));
mean_acc = mean(tabBAcc_3d,2) ;
[~, index] = sort(mean_acc) ;
tabSubjectOrderBAcc = tabSubject(index) ;
[eigenVectors, mu, sigma, mean_, scores, explained] = pcaPythonLike(tabMasks, 2, 0) ;
outTxt = [] ;
disp(outTxt) ;
fprintf(fid, [outTxt '\n']);

%scatter((scores(:,1)-mu)/sigma,(scores(:,2)-mu)/sigma)
dx = (max(scores(:))-min(scores(:))) / 1000 ;
text((scores(index,1)-mu)/sigma+dx, (scores(index,2)-mu)/sigma+dx, num2cell(index), 'fontsize',18);
set(gca, 'fontsize',18); % 20 ticks
xlabel('Interpretation - PCA 1')
ylabel('Interpretation - PCA 2')
%title('PCA on interpretations')
axis([-2.5 2.5 -2.5 2.5])
grid on;
saveas(gcf,['./out/PCA_space'],'epsc')
% close all;

figure;
plot(explained,'-o');
xlabel('Number of PCA dimensions')
ylabel('Explained variance')
grid on;
set(gca, 'fontsize',18); % 20 ticks
XTick_pos    = [1, 2, 3, 5, 10 15 20 25 30] ;
XTick_labels = XTick_pos ;
set(gca, 'XTick', XTick_pos, 'XTickLabel', XTick_labels,'fontsize',18);
saveas(gcf,['./out/explained_variance_nb_dim_PCA'],'epsc')

%% consistency of interpretations
subplot(121);plot(averagedCorr);hold on;
errorbar(mean(tabBAcc_3d,2),std(tabBAcc_3d'));axis('square');
xlabel('Subject #');ylabel('Accuracy (red) / Consistency (blue)');
subplot(122);scatter(averagedCorr,mean(tabBAcc_3d,2));axis([.5 1 .5 1]);axis('square');
xlabel('Consistency (avg pairwise corr btw interpretations)');ylabel('Averaged Balanced Accuracy')
%[r,p] = corr(averagedCorr',mean(tabBAcc_3d,2));

[BF10_SSS_BAcc,r_SSS_BAcc,p_SSS_BAcc] = corrBF(averagedCorr',mean(tabBAcc_3d,2)) ;
outTxt = ['Correlation Consistency/BAcc: BF10=',num2str(BF10_SSS_BAcc),', r=',num2str(r_SSS_BAcc),', p=',num2str(p_SSS_BAcc)];
disp(outTxt) ;
fprintf(fid, [outTxt '\n']);

saveas(gcf,['./out/correlation_consistency_BAcc'],'epsc')
%close all;

%% correlation BAcc / SSS
figure
%scatter(mean(tabBAcc_3d,2),sleepLoss);
axis([.5 1.03 1 7 ]);axis('square');
text(mean_acc(index), sleepLoss(index), num2cell((1:22)),'fontsize',14);
xlabel('Averaged Balanced Accuracy');ylabel('Mean SSS Day 3')
[BF10_SSS_BAcc,r_SSS_BAcc,p_SSS_BAcc] = corrBF(mean(tabBAcc_3d,2),sleepLoss) ;
outTxt = ['Correlation SSS/BAcc: BF10=',num2str(BF10_SSS_BAcc),', r=',num2str(r_SSS_BAcc),', p=',num2str(p_SSS_BAcc)];
set(gca, 'fontsize',18); % 20 ticks
saveas(gcf,['./out/correlation_SSS_BAcc'],'epsc')
disp(outTxt) ;
fprintf(fid, [outTxt '\n']);
%close all;


%% def sub representations Scale-Rate / Frequency-Rate / Frequency-Scale

% full
tabMasks_sub = squeeze(reshape(tabMasks,length(vecSubject),22,8,128)) ;
Ny = 8;
Nx = 22;
Nz = 128;


tabMasks = reshape(tabMasks_sub,length(vecSubject),Nx*Ny*Nz) ;
dim2avg = 3 ;
X = rates;
Y = scales;

%% continuum subjects with real maps PCA_1
[~, i_PCA1] = sort(scores(:,1)) ;
scaleLabels = fliplr([1 2 4 8]) ;
scalePos    = sort(9-[2 4 6 8])*(2^4-1) ;
for iFile = 1:length(vecSubject) 
    load(fileList(index(iFile)).name);
    toPlot = rot90(interp2(mean(reshape(tabMasks(i_PCA1(iFile),:),Nx,Ny,Nz),dim2avg),4)) ;
    XTick_pos    = linspace(1,numrc(toPlot,2),6) ;
    XTick_labels = X(round(linspace(1,length(X),6))) ;
%     YTick_pos    = linspace(1,numrc(toPlot,1),7) ;
%     YTick_labels = fliplr(Y(round(linspace(1,length(Y),7)))) ;
    YTick_pos    = scalePos ;
    YTick_labels = scaleLabels ;    
    plot_sr(toPlot,10*[-1e-3 1e-3], 'Rate (Hz)', 'Scale (cyc/oct)',... 
                       XTick_pos, XTick_labels,...
                       YTick_pos, YTick_labels, 1) ;    
    title(num2str(iSubject));
    saveas(gcf,['./out/ScaleRate/MaskBySubject/',num2str(iFile),'_sub#_',num2str(iSubject)],'epsc')
end
close all;

%% continuum subjects BAcc
figure;
%[~, i_PCA1] = sort(scores(:,1)) ;
% tlo = tiledlayout(3,8) ;

for iFile = 1:length(vecSubject) 
    subplot(3,8,iFile)
    load(fileList(index(iFile)).name);    
    canonicalMap = rot90(mean(reshape(tabMasks(index(iFile),:),Nx,Ny,Nz),dim2avg)) ;
%     IPCA_PL = inversePcaPythonLike([scores(i_PCA1(iFile),1) scores(iFile,2)], eigenVectors, 0, mu, sigma, mean_) ;
%     canonicalMap_reconstruct = rot90(mean(reshape(IPCA_PL(1,:),Nx,Ny,Nz),dim2avg)) ;
%     subplot(121)
%     h(iFile) = nexttile(tlo);
    toPlot = interp2(canonicalMap,4) ;
    h = imagesc(toPlot,[-.01 .01]);

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

saveas(gcf,['./out/ScaleRate/subjectMasks_sorted_from_BAcc'],'epsc')
%close all;


%% continuum subjects PCA_1
figure;
[~, i_PCA1] = sort(scores(:,1)) ;

for iFile = 1:length(vecSubject) 
    subplot(3,8,iFile)    
    load(fileList(i_PCA1(iFile)).name);    
    canonicalMap = rot90(mean(reshape(tabMasks(i_PCA1(iFile),:),Nx,Ny,Nz),dim2avg)) ;
%     IPCA_PL = inversePcaPythonLike([scores(i_PCA1(iFile),1) scores(iFile,2)], eigenVectors, 0, mu, sigma, mean_) ;
%     canonicalMap_reconstruct = rot90(mean(reshape(IPCA_PL(1,:),Nx,Ny,Nz),dim2avg)) ;
%     subplot(121)
    imagesc(canonicalMap,[-.01 .01]);
    axis('square')

%     subplot(122)
%     imagesc(canonicalMap_reconstruct,[-.01 .01]) ;
    title(num2str(iSubject)) ;
    colormap(customcolormap_preset('red-white-blue'));
end
saveas(gcf,['./out/ScaleRate/subjectMasks_sorted_from_PC1'],'epsc')
close all;

%% continuum with idealized masks PCA1
continuum_PC1 = [linspace(-2, 2, 30)' zeros(30,1)] ;
continuum_PC2 = [zeros(30,1) linspace(-2, 2, 30)'] ;
[output_PC1] = inversePcaPythonLike(continuum_PC1, eigenVectors, 0, mu, sigma, mean_) ;
[output_PC2] = inversePcaPythonLike(continuum_PC2, eigenVectors, 0, mu, sigma, mean_) ;

fig = figure;
% filename = 'PC1.gif' ;
for iFile = 1:length(continuum_PC1)
    subplot(4,8,iFile)
    toPlot_PC1 = squeeze(mean(reshape(output_PC1(iFile,:),Nx,Ny,Nz),dim2avg)) ;
    imagesc(interp2(rot90(toPlot_PC1),4),2*[-9e-3 9e-3])
    axis('square')
    
    title(num2str(continuum_PC1(iFile,1)))
    %colorbar
    colormap(customcolormap_preset('red-white-blue'));
%     frame = getframe(fig) ;
%     [A,map] = rgb2ind(frame2im(frame),256);
%     if iFile == 1
%         imwrite(A,map,filename,'gif','LoopCount',Inf,'DelayTime',.1);
%     else
%         imwrite(A,map,filename,'gif','WriteMode','append','DelayTime',.1);
%     end    
end
saveas(gcf,['./out/ScaleRate/idealizedPCA_PC1'],'epsc')
close all;

%% continuum with idealized masks PCA2
fig = figure;

% filename = 'PC2.gif' ;
for iFile = 1:length(continuum_PC2)
    subplot(4,8,iFile)
    toPlot_PC2 = squeeze(mean(reshape(output_PC2(iFile,:),Nx,Ny,Nz),dim2avg)) ;
    imagesc(interp2(rot90(toPlot_PC2),4),1*[-9e-3 9e-3])
    title(num2str(continuum_PC2(iFile,2)))
    axis('square')

    colormap(customcolormap_preset('red-white-blue'));
%     frame = getframe(fig) ;
%     [A,map] = rgb2ind(frame2im(frame),256);
%     if iFile == 1
%         imwrite(A,map,filename,'gif','LoopCount',Inf,'DelayTime',.1);
%     else
%         imwrite(A,map,filename,'gif','WriteMode','append','DelayTime',.1);
%     end     
end
saveas(gcf,['./out/ScaleRate/idealizedPCA_PC2'],'epsc')

%% Variance of idealized PCA
PC1 = squeeze(var(squeeze(mean(reshape(output_PC1(:,:),30,Nx,Ny,Nz),dim2avg+1)),1)) ;
PC2 = squeeze(var(squeeze(mean(reshape(output_PC2(:,:),30,Nx,Ny,Nz),dim2avg+1)),1)) ;

toPlot = interp2(rot90(PC1),4) ;
numrc(toPlot,2) ;
XTick_pos = linspace(1,numrc(toPlot,2),6) ;
XTick_labels = X(round(linspace(1,length(X),6))) ;

% YTick_pos = linspace(1,numrc(toPlot,1),7) ;
% YTick_labels = fliplr(Y(round(linspace(1,length(Y),7)))) ;
YTick_pos    = scalePos ;
YTick_labels = scaleLabels ; 
range = max(toPlot(:)) ;
plot_sr(toPlot,range*[-1 1], 'Rate (Hz)', 'Scale (cyc/oct)',... 
                   XTick_pos, XTick_labels,...
                   YTick_pos, YTick_labels, 0) ; 
title('Interpretation PC1 - Variance') ;
saveas(gcf,['./out/ScaleRate/idealizedPCA_variance_PC1'],'epsc')               

toPlot = interp2(rot90(PC2),4) ;
range = max(toPlot(:)) ;

numrc(toPlot,2) ;
XTick_pos = linspace(1,numrc(toPlot,2),6) ;
XTick_labels = X(round(linspace(1,length(X),6))) ;
% YTick_pos = linspace(1,numrc(toPlot,1),7) ;
% YTick_labels = fliplr(Y(round(linspace(1,length(Y),7)))) ;
YTick_pos    = scalePos ;
YTick_labels = scaleLabels ; 
plot_sr(toPlot,range*[-1 1], 'Rate (Hz)', 'Scale (cyc/oct)',... 
                   XTick_pos, XTick_labels,...
                   YTick_pos, YTick_labels, 0) ;
colormap(customcolormap_preset('red-white-blue'));
title('Interpretation PC2 - Variance') ;
saveas(gcf,['./out/ScaleRate/idealizedPCA_variance_PC2'],'epsc')


%% Marginals of idealized PCA PCs
figure;
%subplot(121);
toPlot_PC1 = interp2(rot90(PC1),4) ;
toPlot_PC2 = interp2(rot90(PC2),4) ;
XTick_pos = linspace(1,numrc(toPlot,2),6) ;
XTick_labels = X(round(linspace(1,length(X),6))) ;
plot(mean(toPlot_PC1)./max(mean(toPlot_PC1)));hold on;
plot(mean(toPlot_PC2)./max(mean(toPlot_PC2)));
set(gca, 'XTick', XTick_pos, 'XTickLabel', XTick_labels);
xlabel('Rate (Hz)')    
ylabel('Normalized variance')
grid on;
grid(gca,'minor')
legend('PC1','PC2')
axis('square')
saveas(gcf,['./out/ScaleRate/idealizedPCA_variance_marginals_rate'],'epsc')
%close all;

figure;
toPlot_PC1 = interp2(rot90(PC1),4) ;
toPlot_PC2 = interp2(rot90(PC2),4) ;
XTick_pos = linspace(1,numrc(toPlot,1),length(Y)) ;
XTick_labels = (Y(round(linspace(1,length(Y),length(Y))))) ;
plot(fliplr(mean(toPlot_PC1'))./max(mean(toPlot_PC1')));hold on;
plot(fliplr(mean(toPlot_PC2'))./max(mean(toPlot_PC2')));
set(gca, 'XTick', XTick_pos, 'XTickLabel', XTick_labels);
xlabel('Scale (cyc/oct)')  
ylabel('Normalized variance')
grid on;
grid(gca,'minor')
legend('PC1','PC2')
axis('square')
saveas(gcf,['./out/ScaleRate/idealizedPCA_variance_marginals_scale'],'epsc')
%close all;

%% correlation between SSS and PCA Principal Components
[BF10_pc1,r_pc1,p_pc1] = corrBF(sleepLoss,scores(:,1));
[BF10_pc2,r_pc2,p_pc2] = corrBF(sleepLoss,scores(:,2));
outTxt = ['Correlation SSS/PC1: BF10=',num2str(BF10_pc1),', r=',num2str(r_pc1),', p=',num2str(p_pc1)] ;
disp(outTxt) ;
fprintf(fid, [outTxt '\n']);
outTxt = ['Correlation SSS/PC2: BF10=',num2str(BF10_pc2),', r=',num2str(r_pc2),', p=',num2str(p_pc2)] ;
disp(outTxt) ;
fprintf(fid, [outTxt '\n']);

figure
%scatter(sleepLoss,scores(:,1))
coeff__ = polyfit(scores(:,1),sleepLoss,1) ;

dx = (max(sleepLoss)-min(sleepLoss)) / 10000*30;
text(scores(index,1)+dx, sleepLoss(index)+dx, num2cell(index),'fontsize',14);
ylabel('Mean SSS Day 3')
xlabel('Interpretation - PCA 1')

hold on;
plot(linspace(-3,3,100),polyval(coeff__,linspace(-3,3,100)),'k')
axis([-3 3 1 7])
axis('square')
set(gca,'fontsize',18)
saveas(gcf,['./out/correlation_SSS_PC1'],'epsc')

figure
%scatter(sleepLoss,scores(:,2))
coeff__ = polyfit(scores(:,2),sleepLoss,1) ;
dx = (max(sleepLoss)-min(sleepLoss)) / 10000*30;
text(scores(index,2)+dx,sleepLoss(index)+dx, num2cell(index),'fontsize',14);
xlabel('Mean SSS Day 3')
ylabel('Interpretation - PCA 2')
set(gca,'fontsize',18)


hold on;
plot(linspace(-3,3,100),polyval(coeff__,linspace(-3,3,100)),'k')
axis([-3 3 1 7 ])
axis('square')

saveas(gcf,['./out/correlation_SSS_PC2'],'epsc')
%close all;

%% linear model cross-validation
nRepeat = 1000 ;
n = length(sleepLoss);
corrTabPearson = [] ;
corrTabSpearman = [] ;
corrTabR2score = [] ;
tabTest = [] ;
tabSL = [] ;

for iRepeat = 1:nRepeat
    rng(iRepeat) ;
    c = cvpartition(n,'HoldOut',0.5);
    idxTrain = training(c,1);
    idxTest = ~idxTrain;
    mdl = fitlm([scores(idxTest,1) scores(idxTest,2)],sleepLoss(idxTest)) ;
    b = mdl.Coefficients.Estimate(1) + mdl.Coefficients.Estimate(2) * scores(idxTest,1) + mdl.Coefficients.Estimate(3) * scores(idxTest,2);
%     mdl = fitlm([scores(idxTest,2)],sleepLoss(idxTest)) ;
%     b = mdl.Coefficients.Estimate(1) + mdl.Coefficients.Estimate(2) * scores(idxTest,2) ;
    rrrR2score = r2_score(b, sleepLoss(idxTest)) ;
    tabTest = [tabTest; b] ;
    tabSL = [tabSL; sleepLoss(idxTest)] ;
    corrTabR2score = [corrTabR2score rrrR2score] ;
end

[~,p_r2] = ttest(corrTabR2score,0) ;
outTxt = ['Linear model: R2-score: M=',num2str(mean(corrTabR2score)),', SD=',num2str(std(corrTabR2score)),'; t-test vs. 0: df=',num2str(length(corrTabR2score)-1),', p=',num2str(p_r2)] ;
disp(outTxt) ;
fprintf(fid, [outTxt '\n']);
fclose(fid);


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


%% BF10 corr tab


function [r, p, BF10]  = corrBF10_tab(tab)
    sz = size(tab) ;
    r = zeros(sz(1),sz(1)) ;
    p = zeros(sz(1),sz(1)) ;
    BF10 = zeros(sz(1),sz(1)) ;
    for i = (1:sz(1))
        for j = (1:sz(1))
            [BF10_SSS_BAcc,r_SSS_BAcc,p_SSS_BAcc]...
                = corrBF(tab(i,:)',tab(j,:)') ;
            if isinf(BF10_SSS_BAcc)
                BF10_SSS_BAcc = 1000 ;
            end
            r(i,j)    = r_SSS_BAcc ;
            p(i,j)    = p_SSS_BAcc ;
            BF10(i,j) = BF10_SSS_BAcc ;
        end
    end
end

