# ConnectionStrangerDMN

This repository contains the analysis code and data for the paper (under review):  
**“Making new connections: An fNIRS machine learning classification study of neural synchrony in the default mode network”**  
by Grace Qiyuan Miao\*, Ian J. Lieberman\*, Ashley Binnquist, Agnieszka Pluta, Bear M. Goldstein, Rick Dale, and Matthew D. Lieberman.  

## Data Files
- `CC_behavNeuro_70dyads.csv`: Behavioral and fNIRS neuroimaging data for 70 dyads.
- `CC_behav_105dyads.csv`: Behavioral-only data for an additional 35 dyads (total = 105 dyads).

## Scripts
- `CC_analysis_cleaned.m`: Main analysis script for the paper.
- `basic_predict.m`: Custom classification function used for logistic regression and permutation-based prediction analyses.

## Summary
The project investigates whether neural synchrony in the Default Mode Network (DMN), as measured by fNIRS, can predict feelings of interpersonal connection between strangers. We apply linear regression and machine learning classification (using MATLAB’s `fitclinear`) to behavioral and neuroimaging data, evaluating how well DMN synchrony and conversation depth explain or predict dyadic connection.

## Contact
Correspondence concerning this article should be addressed to Grace Miao (q.miao@ucla.edu) or Matthew Lieberman (lieber@ucla.edu).
