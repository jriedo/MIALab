close all
clear all
clc
r=[1,5,10,50,100];

w_mean=[];
g_mean=[];
v_mean=[];

for file=r

    T=readtable(['results_knn_',num2str(file),'.csv']);

    i_w=[1:3:153];
    i_g=[2:3:153];
    i_v=[3:3:153];

    w=T.DICE(i_w);
    g=T.DICE(i_g);
    v=T.DICE(i_v);
    
    w_mean=[w_mean,mean(w)];
    g_mean=[g_mean,mean(g)];
    v_mean=[v_mean,mean(v)];

end

plot(r,w_mean,'--x','LineWidth',1,'MarkerSize',10)
hold on
plot(r,g_mean,'--x','LineWidth',1,'MarkerSize',10)
plot(r,v_mean,'--x','LineWidth',1,'MarkerSize',10)
xlabel('K')
ylabel('DICE')
legend('white','grey','ventricle','location','best')
grid on
