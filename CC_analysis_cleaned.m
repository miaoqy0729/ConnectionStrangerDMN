%% Study Properties
prefix='CC';
datapath=uigetdir('','Choose Main Data Directory');
addpath(genpath('/Users/gracemiumiu/Desktop/CC_Data_Analysis/NIRS/JNeuroChangeFraming_2025-02'))
numareas=4;

%% Load data
%Read in behavioral values for analysis
behav=readtable(strcat(datapath,filesep,'CC_behav_105dyads.csv'));
behavNeuro=readtable(strcat(datapath,filesep,'CC_behavNeuro_70dyads.csv'));

%% Mean scores
connectMean=nanmean(behav.ConnectionComp_avg);
connectMeanNeuro=nanmean(behavNeuro.ConnectionComp_avg);

mean_condShallow = mean(behav.ConnectionComp_avg(behav.cond==0), 'omitnan');
mean_condDeep = mean(behav.ConnectionComp_avg(behav.cond==1), 'omitnan');

mean_condShallowNeuro = mean(behavNeuro.ConnectionComp_avg(behavNeuro.cond==0), 'omitnan');
mean_condDeepNeuro = mean(behavNeuro.ConnectionComp_avg(behavNeuro.cond==1), 'omitnan');

mean_feltShallow = mean(behavNeuro.ConnectionComp_avg(behavNeuro.feltDepth_High1Low0==0), 'omitnan');
mean_feltDeep = mean(behavNeuro.ConnectionComp_avg(behavNeuro.feltDepth_High1Low0==1), 'omitnan');

mean_feltDepth_highConn = mean(behavNeuro.feltDepth_avg(behavNeuro.Connection_High1Low0==1), 'omitnan');
mean_feltDepth_lowConn = mean(behavNeuro.feltDepth_avg(behavNeuro.Connection_High1Low0==0), 'omitnan');

mean_DMNsync_highConn = mean(behavNeuro.DMNcnv(behavNeuro.Connection_High1Low0==1), 'omitnan');
mean_DMNsync_lowConn = mean(behavNeuro.DMNcnv(behavNeuro.Connection_High1Low0==0), 'omitnan'); 

% Standard errors for figures
shallow_vals = behav.ConnectionComp_avg(behav.cond == 0);
deep_vals = behav.ConnectionComp_avg(behav.cond == 1);
se_condShallow = std(shallow_vals) / sqrt(length(shallow_vals));
se_condDeep = std(deep_vals) / sqrt(length(deep_vals));

feltDepth_highConn_vals = behavNeuro.feltDepth_avg(behavNeuro.Connection_High1Low0==1);
feltDepth_lowConn_vals = behavNeuro.feltDepth_avg(behavNeuro.Connection_High1Low0==0);
se_feltDepth_highConn = std(feltDepth_highConn_vals) / sqrt(length(feltDepth_highConn_vals));
se_feltDepth_lowConn = std(feltDepth_lowConn_vals) / sqrt(length(feltDepth_lowConn_vals));

DMNsync_highConn_vals = behavNeuro.DMNcnv(behavNeuro.Connection_High1Low0==1);
DMNsync_lowConn_vals = behavNeuro.DMNcnv(behavNeuro.Connection_High1Low0==0);
DMNsync_highConn_vals = DMNsync_highConn_vals(~isnan(DMNsync_highConn_vals));
DMNsync_lowConn_vals = DMNsync_lowConn_vals(~isnan(DMNsync_lowConn_vals));
se_DMNsync_highConn = std(DMNsync_highConn_vals) / sqrt(length(DMNsync_highConn_vals));
se_DMNsync_lowConn = std(DMNsync_lowConn_vals) / sqrt(length(DMNsync_lowConn_vals));


%% t-tests

% One-sample t-test
[~,p,CI,stats]=ttest(behav.ConnectionComp_avg,3) %whole sample
[~,p,CI,stats]=ttest(behavNeuro.ConnectionComp_avg,3) %Neuro sample

% Two-sample comparison of Condition x Connection
% All dyads
[~,p,CI,stats]=ttest2(behav.ConnectionComp_avg(behav.cond==0),behav.ConnectionComp_avg(behav.cond==1)) 
% Neuro dyads only
[~,p,CI,stats]=ttest2(behavNeuro.ConnectionComp_avg(behavNeuro.cond==0),behavNeuro.ConnectionComp_avg(behavNeuro.cond==1)) 

% Two-sample comparison of Connection across FeltDepth High Low
% Neuro dyads only
[~,p,CI,stats]=ttest2(behavNeuro.ConnectionComp_avg(behavNeuro.feltDepth_High1Low0==0),behavNeuro.ConnectionComp_avg(behavNeuro.feltDepth_High1Low0==1)) 

% Two-sample comparison of FeltDepth across Connection High Low 
% Neuro dyads only
[~,p,CI,stats]=ttest2(behavNeuro.feltDepth_avg(behavNeuro.Connection_High1Low0==0),behavNeuro.feltDepth_avg(behavNeuro.Connection_High1Low0==1)) 

% Two-sample comparison of DMN syn across Connection High Low 
% Neuro dyads only
[~,p,CI,stats]=ttest2(behavNeuro.DMNcnv(behavNeuro.Connection_High1Low0==0),behavNeuro.DMNcnv(behavNeuro.Connection_High1Low0==1)) 


%% Linear regression

% Connection ~ Networks
fitlm(behavNeuro,'ConnectionComp_avg~DMNcnv')
fitlm(behavNeuro,'ConnectionComp_avg~FPNcnv')
fitlm(behavNeuro,'ConnectionComp_avg~VATcnv')
fitlm(behavNeuro,'ConnectionComp_avg~DATcnv')

% condition assignment ~ DMN sync
mdl = fitglm(behavNeuro, 'cond ~ DMNcnv', 'Distribution', 'binomial');
mdl.Coefficients

%felt depth ~ DMN sync
fitlm(behavNeuro,'feltDepth_avg~DMNcnv')

% Connection ~ Areas
fitlm(behavNeuro,'ConnectionComp_avg~mPFC_less')
fitlm(behavNeuro,'ConnectionComp_avg~R_TPJ')
fitlm(behavNeuro,'ConnectionComp_avg~L_TPJ')


%% Multiple linear regression
behavNeuro.cond=categorical(behavNeuro.cond); %Make condition categorical
fitlm(behavNeuro,'ConnectionComp_avg~DMNcnv+cond')
fitlm(behavNeuro,'ConnectionComp_avg~DMNcnv+feltDepth_avg')


%% Split by condition
behavNeuro.cond=categorical(behavNeuro.cond); %Make condition categorical
behavSH=behavNeuro(behavNeuro.cond=="0",:);
behavDP=behavNeuro(behavNeuro.cond=="1",:);

% Sig. 
[r,p]=corrcoef(behavDP.ConnectionComp_avg,behavDP.DMNcnv,'Rows','complete') %Deep sample
[r,p]=corrcoef(behavDP.ConnectionComp_avg,behavDP.FPNcnv,'Rows','complete') %Deep sample
[r,p]=corrcoef(behavDP.ConnectionComp_avg,behavDP.VATcnv,'Rows','complete') %Deep sample

% No sig. 
[r,p]=corrcoef(behavSH.ConnectionComp_avg,behavSH.DMNcnv,'Rows','complete') %Shallow sample
[r,p]=corrcoef(behavSH.ConnectionComp_avg,behavSH.FPNcnv,'Rows','complete') %Shallow sample
[r,p]=corrcoef(behavSH.ConnectionComp_avg,behavSH.VATcnv,'Rows','complete') %Shallow sample




%% Classification 

% Define variables for classification function
cv_strategy=5;
numreps=1000;
balance=false;
a=clock; rng(a(6));

% First, DMN + cond
behavNeuro.cond=double(behavNeuro.cond); %Make condition numerical
X = behavNeuro(:,[3,8]); % X: condition, DMN synchrony
y = logical(table2array(behavNeuro(:, 7))); %All 70 dyads into 2 groups (1 = High Connection, 0 = Low Connection)

%Then, DMN + feltDepth
X = behavNeuro(:,[4,8]); % X: feltDepth, DMN synchrony
y = logical(table2array(behavNeuro(:, 7)));

% Then, DMN alone
X = behavNeuro(:,8); % X: DMN synchrony
y = logical(table2array(behavNeuro(:, 7)));

% Then, mPFC, rTPJ, and lTPJ 
X = behavNeuro(:,[14,15,16]); % mPFC, lTPJ, rTPJ
y = logical(table2array(behavNeuro(:, 7)));

% mPFC alone 
X = behavNeuro(:,14); % X: mPFC
y = logical(table2array(behavNeuro(:, 7)));

% lTPJ alone 
X = behavNeuro(:,15); % X: lTPJ
y = logical(table2array(behavNeuro(:, 7)));

% rTPJ alone 
X = behavNeuro(:,16); % X: rTPJ
y = logical(table2array(behavNeuro(:, 7)));


% Run classification function
warning('off', 'all');
null=false;
[mean_accuracy, mean_dev, all_fold_accuracies] = basic_predict(X, y, 'Logistic Regression', cv_strategy, numreps, balance, null);

% Run with null to get p value 
null=true;
y_connectRandi=y(randperm(length(y)));
[meanAccNull, meanDevNull, null_accuracies] = basic_predict(X,y_connectRandi, 'Logistic Regression', cv_strategy, numreps, balance, null);
null_accuracies=sort(null_accuracies);
pVal=(sum(null_accuracies > mean_accuracy))/numreps;
disp(['The p-value from permutation is: ', num2str(pVal)]);




