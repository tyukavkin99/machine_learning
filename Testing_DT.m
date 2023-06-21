%% Loadinng the model
load final_DT_hyper2.mat
%% Loading the test data
X_test_under = readmatrix('X_test_under.csv');
y_test_under = readmatrix('y_test_under.csv');

%% Testing the model
labelsDT_hyper_2 = predict(ModelDT_hyper_2, X_test_under);

%% Confusion matrix

cm = confusionchart(y_test_under,labelsDT_hyper_2)
cm.NormalizedValues

%% ROC Curve

[X_plot_hyper_2,Y_plot_hyper_2,T_plot_hyper_2,AUC_plot_hyper_2] = perfcurve(y_test_under,... 
    labelsDT_hyper_2, 1);

plot(X_plot_hyper_2,Y_plot_hyper_2,'LineWidth',4)

%% AUC
AUC_plot_hyper_2