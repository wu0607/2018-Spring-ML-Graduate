% Heart Disease Detector
%%Load a training dataset that will be used to estimate p_H, p_{S|H}, p_{C|H}, f_{X|H}
load heartdatasetTraining

%Estimate p_H, p_{S|H}, p_{C|H} as the empirical pmfs of 218 patients.

% Estimate the pmf of H. We get p_H(1)=0.4633
P_H0 = sum(heart_disease==0)/length(heart_disease) %p_H0=P(H=0)
P_H1 = sum(heart_disease==1)/length(heart_disease) %p_H1=P(H=1)

% Estimate the conditional pmf of S given H
P_S_H0 = zeros(2,1); %P_S_H0(s)=P(S=s|H=0)
P_S_H1 = zeros(2,1); %P_S_H1(s)=P(S=s|H=1)
for ind_S =1:2
    P_S_H0(ind_S) = sum(sex(heart_disease==0)==ind_S)/length(sex(heart_disease==0));
    P_S_H1(ind_S) = sum(sex(heart_disease==1)==ind_S)/length(sex(heart_disease==1));
end

% Estimate the conditional pmf of C given H
P_C_H0 = zeros(4,1); %P_C_H0(c)=P(C=c|H=0)
P_C_H1 = zeros(4,1); %P_C_H1(c)=P(C=c|H=1)
for ind_C =1:4
    P_C_H0(ind_C) = sum(chest_pain(heart_disease==0)==ind_C)/length(chest_pain(heart_disease==0));
    P_C_H1(ind_C) = sum(chest_pain(heart_disease==1)==ind_C)/length(chest_pain(heart_disease==1));
end

% Estimate the conditional pdf of X given H as a Gaussian distribution with
% the empirical mean and the empirical variance of the 218 patients
% We get mean_X_H1=254; var_X_H1=2047; mean_X_H0=245; var_X_H0=2182;
mean_X_H0= mean(cholesterol(heart_disease==0));
var_X_H0 = var(cholesterol(heart_disease==0));
f_X_H0 = @(x)exp(-(x-mean_X_H0).^2/2/var_X_H0)/sqrt(2*pi*var_X_H0); %f_X_H0(x) = f_{X|H}(x|0)
mean_X_H1= mean(cholesterol(heart_disease==1));
var_X_H1 = var(cholesterol(heart_disease==1));
f_X_H1 = @(x)exp(-(x-mean_X_H1).^2/2/var_X_H1)/sqrt(2*pi*var_X_H1); %f_X_H1(x) = f_{X|H}(x|1)

%%

load heartdatasetTesting
%Write a code to compute the MAP detected results and the corresponding error rate
%for the data in heartdatasetTesting.mat

% MAP_detected = ...
detect = zeros(50,1)
for i=1:50
    %H0
    prob_H0 = P_S_H0(sex_test(i)) * P_C_H0(chest_pain_test(i))*f_X_H0(cholesterol_test(i))*P_H0; 
    %H1
    prob_H1 = P_S_H1(sex_test(i)) * P_C_H1(chest_pain_test(i))*f_X_H1(cholesterol_test(i))*P_H1; 
    
    if prob_H0>prob_H1
        detect(i) = 0;
    end
    if prob_H0<prob_H1
        detect(i) = 1;
    end
end
% Error_rate = ...
correct = 0;
for i=1:50
    if detect(i) == heart_disease_test(i)
        correct = correct +1;
    end
end
error_rate = 1-(correct/50)

