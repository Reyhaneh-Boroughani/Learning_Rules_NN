%% Assignment 3 (LAB2-1)
% % Subtractive normalization Rule
clc
clear all
close all
%% Initialiazing the parameters
load 'lab2_1_data.csv'
u=lab2_1_data;
w=(rand(2,1))*2-1;          %initialized weight vector between [-1 1]
w_old=zeros(2,1);
lr=1e-4;                    %learning rate
threshold=1e-10;            %threshold on weight change for learning termination
epoch=100000;               %maximum number of epochs for learning termination
iter=1;                     %parameter of while loop
Nu=size(u,1);
n=ones(1,Nu);
%% Implement the basic Hebb rule
while ((norm(w-w_old)>threshold) && (iter<=epoch))      
w_old=w;
w_evolution(:,iter)=w;     %keeps track of changes in weight vector during learning

Shuffled_u= u(:,randperm(size(u,2)));       %shuffling the input data sets 
    for i=1:100
        v=w_old'*Shuffled_u(:,i);           %linear firing rate model
        dw=(v.*Shuffled_u(:,i))-((v*(n*Shuffled_u(:,i))*n')./Nu);  %Subtractive normalization    
        w=w_old+lr*dw;                      %updates the weight 
    end
iter=iter+1;
end
max_number_epoch=iter-1;
%% compute the input correlation matrix(Q) & principal eigenvector of Q
Q=corr(u');                             %input correlation matrix
PC=eigs(Q) ;                            %eigenvectors of Q
%% ploting the training data points, the final weight vector and the principal eigenvector of Q
figure(1);clf
scatter(u(1,:),u(2,:))
hold on
plotv(w)
plot([0 PC(1,1)],[0 PC(2,1)],'k','LineWidth',2);
set(gca,'xlim',[-1.5 1.5],'ylim',[-1.5 1.5])
legend('Training data points','Final weight vector','Principal eigenvector of Q')
title('Subtractive normalization')
% saveas(gcf, 'ScatterPlot_Data.png')
%% plotting the evolution in time of the two components of the w_evolution and norm of w_evolution
iter=1:max_number_epoch;
figure(2);clf
plot(iter,w_evolution(1,:))             
xlabel('time'); ylabel('the first component of w')
title('Subtractive normalization- Evolution of the First Component of w in Time')
axis tight
% saveas(gcf, 'w1_Evolution.png')

figure(3);clf
plot(iter,w_evolution(2,:))            
xlabel('time'); ylabel('the second component of w')
title('Subtractive normalization- Evolution of the Second Component of w in Time')
axis tight
% saveas(gcf, 'w2_Evolution.png')

figure(4);clf
plot(vecnorm(w_evolution))           
xlabel('time'); ylabel('the norm value of w')
title('Subtractive normalization- Evolution w-norm in Time')
axis tight
% saveas(gcf, 'w_norm_Evolution.png')