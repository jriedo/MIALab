clearvars; clearvars -GLOBAL; close all;

ALGOS = {'DF', 'kNN', 'SVM',  'SGD', 'ensemble'};

h = figure();
N=size(ALGOS, 2);
for idx=1:N
    csv = strcat('../results/results_', ALGOS{idx}, '.csv');
    subplot(1, N, idx);
    plot_dice(ALGOS{idx}, csv, idx == 1)
end

set(h, 'PaperUnits', 'inches');
set(h, 'PaperSize', [12 8]);
set(h, 'PaperPositionMode', 'manual');
set(h, 'PaperPosition', [0 0 12 8]);
%set(h, 'renderer', 'painters');
print(h, 'boxplot', '-depsc')

function plot_dice(name, results_file, show_y_label)
    csv = readtable(results_file);

    WM = [];
    GM = [];
    V = [];

    for i=1:size(csv, 1)
       if endsWith(csv(i,:).ID, '-PP')
          if(strcmp(csv(i,:).LABEL,'WhiteMatter'))
              WM = [WM csv(i,:).DICE];
          end
          if(strcmp(csv(i,:).LABEL,'GreyMatter'))
              GM = [GM csv(i,:).DICE];
          end
          if(strcmp(csv(i,:).LABEL,'Ventricles'))
              V = [V csv(i,:).DICE];
          end
       end
    end

    scatter(rand(size(WM))/5+0.48, WM, 14, [0.4, 0.4, 0.4], 'o')
    xlim([0, 2]);
    hold on
    scatter(rand(size(GM))/5+1.48, GM, 14, [0.4, 0.4, 0.4], 'o', 'MarkerEdgeColor',[.7 .7 .7], 'MarkerFaceColor',[.7 .7 .7])
    scatter(rand(size(V))/5+2.48, V, 14, [0.4, 0.4, 0.4], 'o', 'MarkerEdgeColor',[0 0 .8], 'MarkerFaceColor',[0 0 .8])
    boxplot([WM; GM; V]', {'WM', 'GM', 'V'}, 'Symbol', '')
    if show_y_label
        ylabel('Dice Coefficient');
    else
        %set(gca, 'YTickLabel', []);
    end
    ylim([0, 1]);
    xlim([0, 4]);
    title(name)
end


    