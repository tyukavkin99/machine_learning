%% Loading the model
load('final_NB_under.mat')

%% Loading the test data
X_test_under = readmatrix('X_test_under.csv');
y_test_under = readmatrix('y_test_under.csv');

%% Testing the model
labelNBUnder = predict(Model3,X_test_under);

%% Confusion matrix
rng(1)
cm = confusionchart(y_test_under,labelNBUnder);
cm.NormalizedValues

%% ROC Curve
rng(1)
[X,Y,T,AUC] = perfcurve(y_test_under, labelNBUnder, 1);
plot(X,Y,'LineWidth',4)

%% AUC
AUC