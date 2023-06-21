%% Naive Bayes

%% Loading undersampled data

X_under = readmatrix('X_train_under.csv');
y_under = readmatrix('y_train_under.csv');


%% Choosing the hyperparameters for distributions
distNames = {'normal','normal','kernel','mvmn','kernel','mvmn','kernel',...
    'mvmn','mvmn','mvmn','mvmn'};
%% Creating the third model
Model3 = fitcnb(X_under,y_under,...
    "DistributionNames",distNames)

% Checking the third model
rng(1)
CVMdl = crossval(Model3);

% Accuracy
classErr = kfoldLoss(CVMdl,'LossFun','ClassifErr');
accuracy = 1-classErr

%% Saving model 3
save('final_NB_under.mat','Model3')