TITANIC PROJECT NOTES
=====================

Feature Generation
~~~~~~~~~~~~~~~~~~
*  Create new "Cabin Sector" & "Cabin Locale" features from "Cabin" to split cabin into general area & specific cabin number
*  Identify numeric columns -> Impute (fill nans with None)
*  Identify categorical columns -> Impute (fill nans with most frequent)
*  Encode categorical columns -> one hot encoder
*  Tried impute with extension -> did not yield a better result
*  Best result to date is with cat imputer = most frequent, num = mean

*  ToDo
*   Look at other types of encoders like `CountEncoders`, `TargetEncoder` (only on train data),`CatBoost` (target probability calculated from rows above it)
*   Make interaction features by adding two or more categorical columns together (`Itertools` `combination`)
*   Make features look more normalized for linear models and neural networks - sqrt, log, ^n

Feature Selection
~~~~~~~~~~~~~~~~~
* Tried PCA with default values but created a performance drop
* Completed code to display feature importances out of xgboost
* ToDo
    -  Implement features selection functions like `selecKbest` with `fclassif` scorer
    -  Or L1 regularization using `LogisticRegression` with regularization set to 1 or 0.1, and using `SelectFromModel`
    - Why does "Fare" and "Age" show up as most important parameters, but causes overfitting?

Model Training
~~~~~~~~~~~~~~
*  Scale and preprocess features using techniques above
*  Implemented LightGBM model & tuning -> Slight performance boost over random forest
*  Implemented XGBoost with tuning -> Similar result to LightGBM
*  XGB Tuning Parameters
    - parameter is called `num_boost_round` and corresponds to the number of boosting rounds or trees to build
    - `early_stopping_round` when to stop if performance haven’t improved for num rounds specified
    - `max_depth` is the maximum number of nodes allowed from the root to the farthest leaf of a tree
    - `min_child_weight` is the minimum weight (or number of samples if all samples have a weight of 1) required in order to create a new node in the tree
    - `subsample corresponds` to the fraction of observations (the rows) to subsample at each step
    - `colsample_bytree` corresponds to the fraction of features (the columns) to use
    - The `ETA` parameter controls the learning rate
*  Tried LDA and decision tree models but resulted in lower performance
*Todo   - Add steps to retrain on all data as final step
*-

Other ToDos
~~~~~~~~~~~
*   Train on all of the data for the final model with the tuned parameters
*   Look at semisupervised learning
*   Look at adding submission command line wrapper functions

Code Improvement
~~~~~~~~~~~~~~~~
*   Fix docs for sphinx
*   Figure out pylint auto formatter
*   Figure out version control
