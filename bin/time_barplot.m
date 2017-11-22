clearvars; clearvars -GLOBAL; close all;

csv = readtable('../results/runtimes.csv');

% convert to hours
csv.TRAIN_TIME = csv.TRAIN_TIME / 3600;
csv.TEST_TIME = csv.TEST_TIME / 3600;


% test time per ONE sample!
NUM_TEST_SAMPLES=30;

groups = {3};
idx=1;
for s=[3 12 70]
   group =  csv(csv.SIZE==s,:);
   group = sortrows(group, 1);
   groups{idx} = group;
   idx = idx+1;
end

h=figure();
idx=1;
for s=[3 12 70]
    subplot(1,3,idx);
    bars = [groups{idx}.TRAIN_TIME, groups{idx}.TEST_TIME/NUM_TEST_SAMPLES];
    b = bar(bars, 'stacked');
    ylim([0, 1]);
    
    set(gca, 'XTickLabel', (groups{idx}.ALG));
    set(gca,'XTickLabelRotation',45)   
    title(string(s));
    if s == 3
       ylabel('Train / Test Time [h]') 
    end
    if s == 12
        legend(b, 'train', 'test');
    end
    idx = idx+1;
end


set(h, 'PaperUnits', 'inches');
set(h, 'PaperSize', [12 8]);
set(h, 'PaperPositionMode', 'manual');
set(h, 'PaperPosition', [0 0 12 8]);
%set(h, 'renderer', 'painters');
print(h, 'runtimes', '-depsc')
