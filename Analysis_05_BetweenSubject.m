clear all
clearvars ;

% initialize sound path
addpath(genpath('./')); 
load('./out/out_02_classication_population_Level_AllMaps/BetweenSubjects_AllMaps_3d.mat');
fid=fopen('./out/BetweenSubjects/results_between_subjects.txt','w');

% strf parameters
frequencies = 440 * 2 .^ ((-31:96)/24) ;
rates = [-32, -22.6, -16, -11.3, -8, -5.70, -4, -2, -1, -.5, -.25, .25, .5, 1, 2, 4, 5.70, 8, 11.3, 16, 22.6, 32] ;
scales = [0.71, 1.0, 1.41, 2.00, 2.83, 4.00, 5.66, 8.00] ;

Ny = 8 ;
Nx = 22 ;
Nz = 128 ;

outTxt = ['Between subject Balanced Accuracy: M=',num2str(mean(tabBAcc)),', SD=',num2str(std(tabBAcc))] ;
fprintf(fid, [outTxt '\n']) ;
disp(outTxt) ;
fclose(fid);


%% scale-rate
X = rates;
Y = scales;
canonicalMap = nanmean(canonicalAllMaps(:,:),1) ;
dim2avg = 3 ;
toPlot = rot90(interp2(squeeze(mean(reshape(canonicalMap,Nx,Ny,Nz),dim2avg)),4)) ;
XTick_pos    = linspace(1,numrc(toPlot,2),16) ;
XTick_labels = X(round(linspace(1,length(X),16))) ;
YTick_pos    = linspace(1,numrc(toPlot,1),7) ;
YTick_labels = fliplr(Y(round(linspace(1,length(Y),7)))) ;
max_ = max(abs(toPlot(:))) ;
plot_sr(toPlot,1.2*[-max_ max_], 'Scale (cyc/oct)', 'Frequencies (Hz)',... 
                   XTick_pos, XTick_labels,...
                   YTick_pos, YTick_labels, 0) ;    
saveas(gcf,['./out/BetweenSubjects/scale_rate_between_subjects'],'epsc')


%% freq-rate
X = rates;
Y = frequencies;
canonicalMap = nanmean(canonicalAllMaps(:,:),1) ;
dim2avg = 2 ;
toPlot = rot90(interp2(squeeze(mean(reshape(canonicalMap,Nx,Ny,Nz),dim2avg)),4)) ;
XTick_pos    = linspace(1,numrc(toPlot,2),16) ;
XTick_labels = X(round(linspace(1,length(X),16))) ;
YTick_pos    = linspace(1,numrc(toPlot,1),7) ;
YTick_labels = fliplr(Y(round(linspace(1,length(Y),7)))) ;
max_ = max(abs(toPlot(:))) ;
plot_sr(toPlot,1.2*[-max_ max_], 'Rate (Hz)', 'Frequency (Hz)',... 
                   XTick_pos, XTick_labels,...
                   YTick_pos, YTick_labels, 0) ;    
saveas(gcf,['./out/BetweenSubjects/freq_rate_between_subjects'],'epsc')


%% freq-scale
X = scales;
Y = frequencies;
canonicalMap = nanmean(canonicalAllMaps(:,:),1) ;
dim2avg = 1 ;
toPlot = rot90(interp2(squeeze(mean(reshape(canonicalMap,Nx,Ny,Nz),dim2avg)),4)) ;
XTick_pos    = linspace(1,numrc(toPlot,2),16) ;
XTick_labels = X(round(linspace(1,length(X),16))) ;
YTick_pos    = linspace(1,numrc(toPlot,1),7) ;
YTick_labels = fliplr(Y(round(linspace(1,length(Y),7)))) ;
max_ = max(abs(toPlot(:))) ;
plot_sr(toPlot,1.2*[-max_ max_], 'Scale (cyc/oct)', 'Frequency (Hz)',... 
                   XTick_pos, XTick_labels,...
                   YTick_pos, YTick_labels, 0) ;    
saveas(gcf,['./out/BetweenSubjects/freq_scale_between_subjects'],'epsc')

               
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
