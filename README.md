# DecisionTrees

## Functions
Evaluation Functions
- evaluate(test_db, trained_tree) - returns accuracy
- evaluate_cm(test_db, trained_tree) - returns 4x4 confusion matrix
- evaluate_plus(test_db, trained_tree) returns a tuple of confusion matrix, recall array, precision array, f1 array and accuracy, where precision[i] = precision for room i+1

Training Functions
- decision_tree_learning(training_dataset, depth) - returns the trained tree and maximum depth (when called depth should be 0)

Classification Functions
- classify(data, trained_tree) - classifies a single set of measurements, returns the room number

Cross-validation Functions
- cross_validation(dataset, n_folds) - performs n_folds cross-validation on a dataset, returns a tuple of confusion matrix, recall array, precision array, f1 array and accuracy, where precision[i] = precision for room i+1
- nested_cross_validation(dataset, n_folds) - performs n_folds nested cross-validation and pruning on a dataset, returns a tuple of confusion matrix, recall array, precision array, f1 array and accuracy, where precision[i] = precision for room i+1

Pretty Print Functions
- pretty_print(cm, recall, precision, f1, accuracy) - prints out metrics in a nice way
- pretty_print_depth(cm, recall, precision, f1, accuracy, unpruned_depth, pruned_depth) - prints out metrics including depths

## Example Code
To train a tree on a dataset, run the following:
    tree, depth = decision_tree_learning("clean_dataset.txt",0)
    
To evaluate a tree, there are three different functions that can be used, which return different metrics.
To obtain the accuracy run:
```
evaluate(tree, test_set)
```
To obtain the confusion matrix run:
evaluate_cm(tree, test_set)
```
To obtain a tuple of the confusion matrix, recall, precision, f1, accuracy run:
```
evaluate_plus(tree, test_set)
```

To run a simple 10-fold cross-validation and view the results in the terminal use the following code:
```
cm, recall, precision, f1, accuracy = cross_validation("clean_dataset.txt",10)
pretty_print(cm,recall,precision,f1,accuracy)
```
The dataset used can be changed by changing the file path in the nested_cross_validation function.

To run nested 10-fold cross-validation with pruning and view the results in the terminal use the following:
```
cm, recall, precision, f1, accuracy, unpruned_depth, pruned_depth = nested_cross_validation("clean_dataset.txt",10)
pretty_print_depth(cm,recall,precision,f1,accuracy,unpruned_depth,pruned_depth)
```
As before, the dataset used can be changed.
