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
* ToDo
*   Implement features selection functions like `selecKbest` with `fclassif` scorer
*   Or L1 regularization using `LogisticRegression` with regularization set to 1 or 0.1, and using `SelectFromModel`
*   Or `RandomForest` prioritization


Model Training
~~~~~~~~~~~~~~
*  Scale and preprocess features using techniques above
*  Implemented LightGBM model & tuning -> Slight performance boost over random forest
*  Implemented XGBoost with tuning -> Similar result to LightGBM
*  Current best result (mean train 0.826, mean test 0.791, submit 0.778) with

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
