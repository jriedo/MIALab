close all
clear all
clc
tree=[3,4,6,10,20,40,80];
node=[100,200,400,800,1500,2000];

w_mean=[];
g_mean=[];
v_mean=[];
t_plot=[];
n_plot=[];

for t=tree
    for n=node

        T=readtable(['DF_trees_',num2str(t),'_nodes_',num2str(n),'\results.csv']);
        for i=1:length(T.ID)
            id=T.ID(i);
            label=T.LABEL(i);
            if id{1}(length(id{1}))=='P'
               if label{1}(1)=='W'
                   w=T.DICE(i);
               elseif label{1}(1)=='G'
                   g=T.DICE(i);
               elseif label{1}(1)=='V'
                   v=T.DICE(i);
               end   
            end
        end

        w_mean=[w_mean,mean(w)];
        g_mean=[g_mean,mean(g)];
        v_mean=[v_mean,mean(v)];
        t_plot=[t_plot,t];
        n_plot=[n_plot,n];
    end
end

figure(1)
tri=delaunay(t_plot,n_plot);
trisurf(tri,t_plot,n_plot,w_mean)
% l = light('Position',[-50 -15 29]);
lighting phong
shading interp
colorbar EastOutside
xlabel('n trees')
ylabel('n nodes')
zlabel('DICE')
title('white matter')

figure(2)
tri=delaunay(t_plot,n_plot);
trisurf(tri,t_plot,n_plot,g_mean)
% l = light('Position',[-50 -15 29]);
lighting phong
shading interp
colorbar EastOutside
xlabel('n trees')
ylabel('n nodes')
zlabel('DICE')
title('grey matter')

figure(1)
tri=delaunay(t_plot,n_plot);
trisurf(tri,t_plot,n_plot,v_mean)
% l = light('Position',[-50 -15 29]);
lighting phong
shading interp
colorbar EastOutside
xlabel('n trees')
ylabel('n nodes')
zlabel('DICE')
title('ventricles')

