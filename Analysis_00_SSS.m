%% session

close all
load('./out/out_02_classication_population_Level_AllMaps/BetweenSubjects_AllMaps_3d.mat') ;
% nbSamples = [143, 132, 131, 99, 133, 115, 123, 137, 126, 125, 106, 98, 107, 107, 107, 106, 194, 147, 113, 131, 103, 137] ;
mean_acc_btw = mean(tabBAcc,2) ;
std_acc_btw = std(tabBAcc,[],2) ;

load('./out/out_03_classicationSubjectLevel_AllMaps/BAcc_3D.mat')

mean_acc = mean(tabBAcc_3d,2) ;
std_acc = std(tabBAcc_3d,[],2) ;
[mean_acc, index] = sort(mean_acc) ;
std_acc = std_acc(index) ;

figure;
errorbar((1:22)+1,mean_acc,std_acc,'o-','linewidth',2);
hold on;
errorbar((1),mean_acc_btw,std_acc_btw,'o-','linewidth',2);
hold on;
plot(ones(1,23)*0.5,'k','linewidth',2)
hold on;
errorbar(11,.5,.0,'r');
hold off;
% errorbar(0,0.7856617647058823,.028385114921996496,'Color','k')
addpath('./lib') ;
xlabel('Participant #'); % xlabel
ylabel('Balanced Accuracy'); %ylabel
axis([0 24 0.4 1.05]);
axis('square')
grid on
XTick_pos = (1:23);
XTick_labels = {'avg', (1:22)} ;
set(gca, 'XTick', XTick_pos, 'XTickLabel', XTick_labels); % 10 ticks 
saveas(gcf,['./out/BAcc'],'epsc')
%close all;



