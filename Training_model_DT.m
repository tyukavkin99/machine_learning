%% Loading the undersampled data

X_under = readmatrix('X_train_under.csv');
Y_under = readmatrix('y_train_under.csv');

%% Loading the data

ModelDT_under = fitctree(X_under,y_under);

rng(1)
CVTree_under = crossval(ModelDT_under);
TreeErr_under = kfoldLoss(CVTree_under,'LossFun','classiferror');
accuracyDT_under = 1-TreeErr_under

%% Loading the data

ModelDT_hyper = fitctree(X_under,y_under,'PredictorSelection','curvature');

rng(1)
CVTree_hyper = crossval(ModelDT_hyper);
TreeErr_hyper = kfoldLoss(CVTree_hyper,'LossFun','classiferror');
accuracyDT_hyper = 1-TreeErr_hyper

%% Checking the hyperparameters

ModelDT_hyper_1 = fitctree(X_under,y_under,'OptimizeHyperparameters','auto');

rng(1)
CVTree_hyper_1 = crossval(ModelDT_hyper_1);
TreeErr_hyper_1 = kfoldLoss(CVTree_hyper_1,'LossFun','classiferror');
accuracyDT_hyper_1 = 1-TreeErr_hyper_1

%% Remaking the model with hyperparameters
ModelDT_hyper_2 = fitctree(X_under,y_under,'MinLeafSize',5);

rng(1)
CVTree_hyper_2 = crossval(ModelDT_hyper_2);
TreeErr_hyper_2 = kfoldLoss(CVTree_hyper_2,'LossFun','classiferror');
accuracyDT_hyper_2 = 1-TreeErr_hyper_2
%% View
view(ModelDT_hyper,'mode','graph')
%% Saving the model
save('final_DT_hyper2.mat', 'ModelDT_hyper_2')

