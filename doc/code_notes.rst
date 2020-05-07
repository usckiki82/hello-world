Feature Generation
==================
*   Look at one-hot encoding categorical features since some models may not use it.
*   Change time stamp columns into separate hour, min, day etc.
*   Look at other types of encoders like `CountEncoders`, `TargetEncoder` (only on train data),`CatBoost` (target probability calculated from rows above it)
*   Make interaction features by adding two or more categorical columns together (`Itertools` `combination`)
*   Approaches for NaNs: drop columns, impute (mean), impute with extension (extra column indicating if data field was imputed or not)
*   Add rolling feature counts for timeseries data (e.g. last X days or from beginning) or time since last event
*   Make features look more normalized for linear models and neural networks - sqrt, log, ^n

Feature Selection
=================
*   Implement features selection functions like `selecKbest` with `fclassif` scorer
*   Or L1 regularization using `LogisticRegression` with regularization set to 1 or 0.1, and using `SelectFromModel`
*   Or `RandomForest` prioritization


Test Train Split
================
*   On time series data should split in time order so that future events do not leak into the test data.
*   More ways to account or test for data leakage
*   Verify no target leakage = data included in training that would not be available at time of measurement / prediction
*   Train test contamination by performing preprocessing steps using test or crossval data

Model Types
===========
*   Try XGBoost tuning on `n_estimators` and utilizing 'early_stopping_rounds` with value like 5
*   Try LightGBM models tuning on
*   With pipeline, define preprocessing step as a ColumnTransformer containing numerical and categorical transformers, first imputing then encoding categorical data

Submission
==========
*   Train on all of the data for the final model with the tuned parameters
*   Look at semisupervised learning
*   Look at adding submission command line wrapper function
Code Improvement
================
*   Fix docs for sphinx
*   Figure out pylint auto formatter
*   Figure out version control

Computer Vision
===============
*   Tensor = matrix (possibly multidimensional), Convolution = kernel or patch
*   Kaggle has some pretrained models you can get in your workspace
*   ResNet50, V66
*   Transfer learning techniques
*   Use `Sequential` and `Dense` to add a new last layer to a model.
*   Pooling avg refers to taking all of the channels that go to a node and averaging them.
*   Use softmax activation to turn results into probabilities
*   Compile command:  `categorical_crossentropy` = log loss function, optimizer `sgd` = stochastic gradient descent
*   Perform data augmentation by flipping pictures, and shifting by 20% in horizontal/vertical direction
*   Optimization function `adam` determines best learning curve for you
*   `ReLU activation function`
*   One hot encoding of target values
*   Models perform better with a Dense layer between flatten layer and output layer
*   Model layers:  1st Conv with raw data, n_Conv, flatten layer, dense layer, final dense layer that converts to predictions
*   Stride length helps to reduce calculation time -- # of pixels move the kernel by; other option is max pooling but stride length works for more complex models (like generative)
*   Drop out layers used to ignore/remove random sampled nodes from some of the training - helps so node doesn't dominate and overfit