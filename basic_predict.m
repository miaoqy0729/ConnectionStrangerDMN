
function [mean_accuracy, mean_dev, all_fold_accuracies] = basic_predict(X, y, model_name, cv_strategy, numreps, balance, null)
    all_fold_accuracies = [];
    mean_stds = [];
    
    for i = 1:numreps
        fold_results = [];
        if null
            y_new = y(randperm(length(y)));
        else
            y_new = y;
        end
        
        % Generate cv partitions
        if ischar(cv_strategy) && strcmpi(cv_strategy, 'loocv')
            outer_cv = cvpartition(y_new, 'LeaveOut');
        elseif ischar(cv_strategy) && strcmpi(cv_strategy, 'ltocv')
            cv_strategy = length(y_new) / 2;
            outer_cv = cvpartition(y_new, 'KFold', cv_strategy, 'Stratify', true);
        elseif isnumeric(cv_strategy)
            outer_cv = cvpartition(y_new, 'KFold', cv_strategy, 'Stratify', true);
        else
            error("Invalid cv_strategy.");
        end

        parfor j = 1:outer_cv.NumTestSets
            trainIdx = outer_cv.training(j);
            testIdx = outer_cv.test(j);
            X_train = table2array(X(trainIdx, :));
            y_train = y_new(trainIdx);
            X_test = table2array(X(testIdx, :));
            y_test = y_new(testIdx);
            
            % Run model
            model = fitclinear(X_train, y_train, 'Learner', 'logistic');

            % Compute predictions
            y_pred = predict(model, X_test);
            fold_accuracy = mean(y_pred == y_test) * 100;
            
            % Store accuracy
            fold_results = [fold_results, fold_accuracy];
        end
        
        all_fold_accuracies = [all_fold_accuracies, mean(fold_results)];
        mean_stds = [mean_stds, std(fold_results)];
    end
    
    mean_accuracy = mean(all_fold_accuracies);
    mean_dev = mean(mean_stds);

    fprintf('Basic %s | Mean accuracy over %d repetitions: %.3f%% (%.3f%%)\n', model_name, numreps, mean_accuracy, mean_dev);

end
