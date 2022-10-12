clear all
clearvars ;
clc;
% initialize sound path
addpath(genpath('./')); 
fileList = dir(strcat('./out/out_03_classicationSubjectLevel_AllMaps/*_3d.mat')) ;
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

load('./out/out_03_classicationSubjectLevel_AllMaps/BAcc_3D.mat'); % load balanced accuracies
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
    stanfordAllMaps = [stanfordAllMaps; repmat(SSS(iFile,:),[N_seed 1])] ;
    triu_ = triu(corr(canonicalAllMaps(end-length(canonicalAllMaps(:,1))+1:end,:)'),1) ;
    triu_(triu_==0) = [] ;
    averagedCorr(iFile) = nanmean(triu_) ;
end

%% consistency of interpretations
subplot(121);
plot(averagedCorr);hold on;
errorbar(mean(tabBAcc_3d,2),std(tabBAcc_3d'));axis('square');
xlabel('Subject #');ylabel('Accuracy (red) / Consistency (blue)');
subplot(122);scatter(averagedCorr,mean(tabBAcc_3d,2));axis([.5 1 .5 1]);axis('square');
xlabel('Consistency (avg pairwise corr btw interpretations)');ylabel('Averaged Balanced Accuracy')

[BF10_Consistency_BAcc,r_Consistency_BAcc,p_Consistency_BAcc] = corrBF(averagedCorr',mean(tabBAcc_3d,2)) ;
outTxt = ['Correlation Consistency/BAcc: BF10=',num2str(BF10_Consistency_BAcc),', r=',num2str(r_Consistency_BAcc),', p=',num2str(p_Consistency_BAcc)];
disp(outTxt) ;
fprintf(fid, [outTxt '\n']);
saveas(gcf,['./out/correlation_consistency_BAcc'],'epsc')
close all;

%% correlation BAcc / SSS
scatter(mean(tabBAcc_3d,2),sleepLoss);axis([.5 1 1 7]);axis('square');
xlabel('Averaged Balanced Accuracy');ylabel('Sleepiness (Stanford Scale)')
[BF10_SSS_BAcc,r_SSS_BAcc,p_SSS_BAcc] = corrBF(mean(tabBAcc_3d,2),sleepLoss) ;
outTxt = ['Correlation SSS/BAcc: BF10=',num2str(BF10_SSS_BAcc),', r=',num2str(r_SSS_BAcc),', p=',num2str(p_SSS_BAcc)];
disp(outTxt) ;
fprintf(fid, [outTxt '\n']);
saveas(gcf,['./out/correlation_SSS_BAcc'],'epsc')
close all;

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

%% PCA on Masks
figure
load('./out/out_03_classicationSubjectLevel_AllMaps/BAcc_3D.mat');
mean_acc = mean(tabBAcc_3d,2) ;
[~, index] = sort(mean_acc) ;
tabSubjectOrderBAcc = tabSubject(index) ;
[eigenVectors, mu, sigma, mean_, scores, explained] = pcaPythonLike(tabMasks, 2, 0) ;
outTxt = [] ;
disp(outTxt) ;
fprintf(fid, [outTxt '\n']);

%scatter((scores(:,1)-mu)/sigma,(scores(:,2)-mu)/sigma)
dx = (max(scores(:))-min(scores(:))) / 1000 ;
text((scores(:,1)-mu)/sigma+dx, (scores(:,2)-mu)/sigma+dx, num2cell(tabSubjectOrderBAcc));
xlabel('PC1')
ylabel('PC2')
%title('PCA on interpretations')
axis([-2.5 2.5 -2 2])
grid on;
saveas(gcf,['./out/PCA_space'],'epsc')
% close all;

%% continuum subjects with real maps PCA_1
[~, i_PCA1] = sort(scores(:,1)) ;

for iFile = 1:length(vecSubject) 
    load(fileList(i_PCA1(iFile)).name);
    toPlot = rot90(interp2(squeeze(mean(reshape(tabMasks(i_PCA1(iFile),:),Nx,Ny,Nz),dim2avg)),4)) ;
    XTick_pos    = linspace(1,numrc(toPlot,2),16) ;
    XTick_labels = X(round(linspace(1,length(X),16))) ;
    YTick_pos    = linspace(1,numrc(toPlot,1),7) ;
    YTick_labels = fliplr(Y(round(linspace(1,length(Y),7)))) ;
    plot_sr(toPlot,20*[-1e-3 1e-3], 'Rate (Hz)', 'Frequency (Hz)',... 
                       XTick_pos, XTick_labels,...
                       YTick_pos, YTick_labels, 1) ;    
    title(num2str(iSubject));
    saveas(gcf,['./out/FreqRate/MaskBySubject/',num2str(iFile),'_sub#_',num2str(iSubject)],'epsc')
end
close all;

%% continuum subjects PCA_1
figure;
[~, i_PCA1] = sort(scores(:,1)) ;

for iFile = 1:length(vecSubject) 
    subplot(3,8,iFile)    
    load(fileList(i_PCA1(iFile)).name);    
    canonicalMap = rot90(interp2(squeeze(mean(reshape(tabMasks(i_PCA1(iFile),:),Nx,Ny,Nz),dim2avg)),4)) ;
%     IPCA_PL = inversePcaPythonLike([scores(i_PCA1(iFile),1) scores(iFile,2)], eigenVectors, 0, mu, sigma, mean_) ;
%     canonicalMap_reconstruct = rot90(mean(reshape(IPCA_PL(1,:),Nx,Ny,Nz),dim2avg)) ;
%     subplot(121)
    imagesc(canonicalMap,20*[-1e-3 1e-3]);

%     subplot(122)
%     imagesc(canonicalMap_reconstruct,[-.01 .01]) ;
    title(num2str(iSubject)) ;
    colormap(customcolormap_preset('red-white-blue'));
end
saveas(gcf,['./out/FreqRate/subjectMasks_sorted_from_PC1'],'epsc')
close all;

%% continuum with idealized masks PCA1
continuum_PC1 = [linspace(-5, 5, 30)' zeros(30,1)] ;
continuum_PC2 = [zeros(30,1) linspace(-5, 5, 30)'] ;
[output_PC1] = inversePcaPythonLike(continuum_PC1, eigenVectors, 0, mu, sigma, mean_) ;
[output_PC2] = inversePcaPythonLike(continuum_PC2, eigenVectors, 0, mu, sigma, mean_) ;

fig = figure;
% filename = 'PC1.gif' ;
for iFile = 1:length(continuum_PC1)
    subplot(4,8,iFile)
    toPlot_PC1 = squeeze(mean(reshape(output_PC1(iFile,:),Nx,Ny,Nz),dim2avg)) ;
    imagesc(interp2(rot90(toPlot_PC1),4),5*[-9e-3 9e-3])
%     axis('square')
    
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
saveas(gcf,['./out/FreqRate/idealizedPCA_PC1'],'epsc')
close all;

%% continuum with idealized masks PCA2
fig = figure;

% filename = 'PC2.gif' ;
for iFile = 1:length(continuum_PC2)
    subplot(4,8,iFile)
    toPlot_PC2 = squeeze(mean(reshape(output_PC2(iFile,:),Nx,Ny,Nz),dim2avg)) ;
    imagesc(interp2(rot90(toPlot_PC2),4),10*[-9e-3 9e-3])
    title(num2str(continuum_PC2(iFile,2)))
%     axis('square')

    colormap(customcolormap_preset('red-white-blue'));
%     frame = getframe(fig) ;
%     [A,map] = rgb2ind(frame2im(frame),256);
%     if iFile == 1
%         imwrite(A,map,filename,'gif','LoopCount',Inf,'DelayTime',.1);
%     else
%         imwrite(A,map,filename,'gif','WriteMode','append','DelayTime',.1);
%     end     
end
saveas(gcf,['./out/FreqRate/idealizedPCA_PC2'],'epsc')

%% Variance of idealized PCA
PC1 = squeeze(var(squeeze(mean(reshape(output_PC1(:,:),30,Nx,Ny,Nz),dim2avg+1)),1)) ;
PC2 = squeeze(var(squeeze(mean(reshape(output_PC2(:,:),30,Nx,Ny,Nz),dim2avg+1)),1)) ;

figure;
toPlot = interp2(rot90(PC1),4) ;
numrc(toPlot,2) ;
XTick_pos = linspace(1,numrc(toPlot,2),6) ;
XTick_labels = X(round(linspace(1,length(X),6))) ;

YTick_pos = linspace(1,numrc(toPlot,1),7) ;
YTick_labels = fliplr(Y(round(linspace(1,length(Y),7)))) ;

plot_sr(toPlot,.2*[-9e-3 9e-3], 'Rate (Hz)', 'Frequency (Hz)',... 
                   XTick_pos, XTick_labels,...
                   YTick_pos, YTick_labels, 1) ;              
saveas(gcf,['./out/FreqRate/idealizedPCA_variance_PC1'],'epsc')
toPlot = interp2(rot90(PC2),4) ;
numrc(toPlot,2) ;
XTick_pos = linspace(1,numrc(toPlot,2),6) ;
XTick_labels = X(round(linspace(1,length(X),6))) ;
YTick_pos = linspace(1,numrc(toPlot,1),7) ;
YTick_labels = fliplr(Y(round(linspace(1,length(Y),7)))) ;
plot_sr(toPlot,.2*[-9e-3 9e-3], 'Rate (Hz)', 'Frequency (Hz)',... 
                   XTick_pos, XTick_labels,...
                   YTick_pos, YTick_labels, 1) ;
colormap(customcolormap_preset('red-white-blue'));
saveas(gcf,['./out/FreqRate/idealizedPCA_variance_PC2'],'epsc')
close all;

%% Marginals of idealized PCA PCs
figure;
%subplot(121);
toPlot_PC1 = interp2(rot90(PC1),4) ;
toPlot_PC2 = interp2(rot90(PC2),4) ;
XTick_pos = linspace(1,numrc(toPlot,2),16) ;
XTick_labels = X(round(linspace(1,length(X),16))) ;
plot(mean(toPlot_PC1)./max(mean(toPlot_PC1)));hold on;
plot(mean(toPlot_PC2)./max(mean(toPlot_PC2)));
set(gca, 'XTick', XTick_pos, 'XTickLabel', XTick_labels);
xlabel('Rate (Hz)')    
ylabel('Normalized variance')
grid on;
grid(gca,'minor')
legend('PC1','PC2')
axis('square')
saveas(gcf,['./out/FreqRate/idealizedPCA_variance_marginals_rate'],'epsc')

figure;
toPlot_PC1 = rot90(PC1) ;
toPlot_PC2 = rot90(PC2) ;

marginal_pc1 = fliplr(mean(toPlot_PC1'))./max(mean(toPlot_PC1')) ;
marginal_pc2 = fliplr(mean(toPlot_PC2'))./max(mean(toPlot_PC2')) ;
toPlot_PC1 = interp(marginal_pc1,4);
toPlot_PC2 = interp(marginal_pc2,4);
XTick_pos = linspace(1,length(toPlot_PC1),8) ;
XTick_labels = (Y(round(linspace(1,length(Y),8)))) ;
plot(toPlot_PC1);hold on;
plot(toPlot_PC2);
set(gca, 'XTick', XTick_pos, 'XTickLabel', XTick_labels);
xlabel('Frequency (Hz)')  
ylabel('Normalized variance')
grid on;
grid(gca,'minor')
legend('PC1','PC2')
axis('square')
save('./out/FreqRate/marginals_frequency_interp.mat','marginal_pc1','marginal_pc2') ;
saveas(gcf,['./out/FreqRate/idealizedPCA_variance_marginals_frequencies'],'epsc')
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
subplot(121)
%scatter(sleepLoss,scores(:,1))
coeff__ = polyfit(sleepLoss,scores(:,1),1) ;

dx = (max(sleepLoss)-min(sleepLoss)) / 10000*30;
text(sleepLoss+dx, scores(:,1)+dx, num2cell(tabSubjectOrderBAcc));
xlabel('Mean SSS Day 3')
ylabel('PC1')

hold on;
plot(linspace(1,7,100),polyval(coeff__,linspace(1,7,100)),'k')
axis([1 7 -3 3])
axis('square')

subplot(122)
%scatter(sleepLoss,scores(:,2))
coeff__ = polyfit(sleepLoss,scores(:,2),1) ;
dx = (max(sleepLoss)-min(sleepLoss)) / 10000*30;
text(sleepLoss+dx, scores(:,2)+dx, num2cell(tabSubjectOrderBAcc));
xlabel('Mean SSS Day 3')
ylabel('PC2')

hold on;
plot(linspace(1,7,100),polyval(coeff__,linspace(1,7,100)),'k')
axis([1 7 -3 3])
axis('square')

saveas(gcf,['./out/correlation_SSS_PCs'],'epsc')
close all;

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

function plot_sr(im, cl, xlabel_, ylabel_, XTick_pos,...
                   XTick_labels,YTick_pos, YTick_labels, show)
    if show==1
        figure('visible','off');
    else
        figure();
    end
       
    imagesc(im, cl);
    xlabel(xlabel_);
    ylabel(ylabel_);
    colorbar;
    set(gca, 'XTick', XTick_pos, 'XTickLabel', XTick_labels); % 10 ticks 
    set(gca, 'YTick', YTick_pos, 'YTickLabel', YTick_labels); % 20 ticks
    colormap(customcolormap_preset('red-white-blue'));
    axis('square')
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
