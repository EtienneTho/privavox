clear all
clearvars ;
clc;

% strf parameters
frequencies = 440 * 2 .^ ((-31:96)/24) ;
rates = [-32, -22.6, -16, -11.3, -8, -5.70, -4, -2, -1, -.5, -.25, .25, .5, 1, 2, 4, 5.70, 8, 11.3, 16, 22.6, 32] ;
scales = [0.71, 1.0, 1.41, 2.00, 2.83, 4.00, 5.66, 8.00] ;

vecSubject = (1:22); 

% full
Ny = 8;
Nx = 22;
Nz = 128;

%% load all data
% STMs => all STMs
% labels => labels for the restriction 0: before, 1: after
% subjectNb => subject number

load('./AcousticAnalysis_BetweenSubjects_all.mat') ;
STMs_pre = STMs(labels==0,:);
STMs_post = STMs(labels==1,:);



% 
% %% within subjects    
% %% mean post freq-rate
% tabMaskAcoustic = [] ;
% maxFR = [] ;
% 
% for iFile = 1:length(vecSubject) 
%     
%     load(fileListWithin(iFile).name) ;
%     % mean_post
%     toPlot_mp = rot90(squeeze(mean(reshape(mean_post,Nx,Ny,Nz),dim2avg))) ;
%     tabMaskAcoustic_meanPost = [tabMaskAcoustic; mean(toPlot,2)'] ;
%     toPlot = interp2(toPlot_mp,4) ;
%     XTick_pos    = linspace(1,numrc(toPlot,2),6) ;
%     XTick_labels = X(round(linspace(1,length(X),6))) ;
%     YTick_pos    = freqPos ;
%     YTick_labels = freqLabels ;
%     plot_sr(toPlot,500*[-1e-3 1e-3], 'Rate (Hz)', 'Frequency (Hz)',... 
%                        XTick_pos, XTick_labels,...
%                        YTick_pos, YTick_labels, 1) ;
%     colormap(customcolormap_preset('red-white-blue'));
% 
%     title(num2str(iSubject));
%     saveas(gcf,['./out/Acoustics/mean_post/freqRate/mean_post_',num2str(iFile),'_sub#_',num2str(iSubject)],'epsc')
%     
%     % mean_pre    
%     toPlot_pr = rot90(squeeze(mean(reshape(mean_pre,Nx,Ny,Nz),dim2avg))) ;
%     tabMaskAcoustic_meanPre = [tabMaskAcoustic; mean(toPlot,2)'] ;
%     toPlot = interp2(toPlot_pr,4) ;    
%     XTick_pos    = linspace(1,numrc(toPlot,2),6) ;
%     XTick_labels = X(round(linspace(1,length(X),6))) ;
% %     YTick_pos    = linspace(1,numrc(toPlot,1),7) ;
% %     YTick_labels = fliplr(Y(round(linspace(1,length(Y),7)))) ;
%     YTick_pos    = freqPos ;
%     YTick_labels = freqLabels ;
%     plot_sr(toPlot,500*[-1e-3 1e-3], 'Rate (Hz)', 'Frequency (Hz)',... 
%                        XTick_pos, XTick_labels,...
%                        YTick_pos, YTick_labels, 1) ;
%     colormap(customcolormap_preset('red-white-blue'));
% 
%     title(num2str(iSubject));
%     saveas(gcf,['./out/Acoustics/mean_pre/freqRate/mean_pre_',num2str(iFile),'_sub#_',num2str(iSubject)],'epsc')
%     
%     % diff_mean
%     toPlot = reshape(2 * abs(toPlot_mp-toPlot_pr) ./ (toPlot_mp+toPlot_pr),Nz,Nx) ;    
%     maxFR = [maxFR max(toPlot(:))] ;
%     tabMaskAcoustic_diffMean = [tabMaskAcoustic; mean(toPlot,2)'] ;
%     toPlot_ = interp2(toPlot,4) ;    
%     XTick_pos    = linspace(1,numrc(toPlot_,2),6) ;
%     XTick_labels = X(round(linspace(1,length(X),6))) ;
% %     YTick_pos    = linspace(1,numrc(toPlot_,1),7) ;
% %     YTick_labels = fliplr(Y(round(linspace(1,length(Y),7)))) ;
%     YTick_pos    = freqPos ;
%     YTick_labels = freqLabels ;
%     plot_sr(toPlot_,500*[-1e-3 1e-3], 'Rate (Hz)', 'Frequency (Hz)',... 
%                        XTick_pos, XTick_labels,...
%                        YTick_pos, YTick_labels, 1) ;
%     colormap(customcolormap_preset('red-white-blue'));
% 
%     title(num2str(iSubject));
%     tabMaskAcoustic_meanPost = [tabMaskAcoustic; mean(toPlot,2)'] ;
%     saveas(gcf,['./out/Acoustics/diff_mean/freqRate/diff_mean',num2str(iFile),'_sub#_',num2str(iSubject)],'epsc')
%     close all    
%     % diff_mean
% end
% 
% 
% %% between
% %% freq-rate
% X = rates;
% Y = frequencies;
% freqLabels = fliplr([250 500 1000 2000 4000]) ;
% freqPos    = sort(128-floor(24*log2(freqLabels/440)+36))*2^4 ;
% dim2avg = 2 ;
% 
% load('./AcousticAnalysis_BetweenSubjects.mat')
% post_freq_rate = rot90(squeeze(mean(reshape(mean_post,Nx,Ny,Nz),dim2avg))) ;
% pre_freq_rate = rot90(squeeze(mean(reshape(mean_pre,Nx,Ny,Nz),dim2avg))) ;    
% diff_mean_freq_rate = reshape(2 * abs(toPlot_mp-toPlot_pr) ./ (toPlot_mp+toPlot_pr),Nz,Nx) ;
% 
% %% scale-rate
% X = rates;
% Y = scales;
% scaleLabels = fliplr([1 2 4 8]) ;
% scalePos    = sort(9-[2 4 6 8])*(2^4-1) ;
% dim2avg     = 3 ;
% 
% load('./AcousticAnalysis_BetweenSubjects.mat')
% post_scale_rate = rot90(squeeze(mean(reshape(mean_post,Nx,Ny,Nz),dim2avg))) ;
% pre_scale_rate = rot90(squeeze(mean(reshape(mean_pre,Nx,Ny,Nz),dim2avg))) ;
% diff_mean_scale_rate = reshape(2 * abs(toPlot_mp-toPlot_pr) ./ (toPlot_mp+toPlot_pr),Ny,Nx) ;

