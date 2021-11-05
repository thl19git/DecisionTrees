# DecisionTrees
Decision Tree CW

Evaluation Functions
- evaluate(test_db,trained_tree) - returns accuracy
- evaluate_cm(test_db,trained_tree) - returns 4x4 confusion matrix
- evaluate_plus(test_db,trained_tree) returns a tuple of confusion matrix, recall array, precision array, f1 array and accuracy, where precision[i] = precision for room i+1

Training Functions
- decision_tree_learning(training_dataset, depth) - returns the trained tree and maximum depth (when called depth should be 0)

Classification Functions
- classify(data, trained_tree) - classifies a single set of measurements, returns the room number

Cross-validation Functions
- cross_validation(dataset,n_folds) - performs n_folds cross-validation on a dataset, returns a tuple of confusion matrix, recall array, precision array, f1 array and accuracy, where precision[i] = precision for room i+1
- nested_cross_validation(dataset,n_fold) - performs n_folds nested cross-validation and pruning on a dataset, returns a tuple of confusion matrix, recall array, precision array, f1 array and accuracy, where precision[i] = precision for room i+1

Pretty Print Functions
- pretty_print(cm,recall,precision,f1,accuracy) - prints out metrics in a nice way
- pretty_print_depth(cm,recall,precision,f1,accuracy,unpruned_depth,pruned_depth) - prints out metrics including depths
