import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import operator
import xgboost as xgb

from sklearn.model_selection import learning_curve


def plot_learning_curve(X, y, estimator, title, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5),
                        scoring=None, shuffle=True):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training samples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring=scoring, shuffle=shuffle)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def plot_roc_curve(fpr, tpr, roc_auc, title_prefix):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.plot(fpr, tpr, lw=2)
    ax.plot([0, 1], [0, 1], linestyle='--', color='k')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid()
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')

    ax.set_title(f'{title_prefix.capitalize()} Receiver Operating Characteristic (ROC_AUC = %0.3f)' % roc_auc)


def plot_feature_histogram(df, feature_cols, class_col="class_label", hist_check_int=False):
    fig, axes = plt.subplots(len(feature_cols), 1, figsize=(10, 3 * len(feature_cols)))
    n_bins = 40
    title_prefix = "Histogram: "

    axes = np.array(axes)  # necessary for cases where there is only one feature col

    for i, ax in enumerate(axes.flatten()):
        is_numeric = pd.to_numeric(df[feature_cols[i]], errors='coerce').notnull().all()

        if is_numeric:
            min_bin = min(df[feature_cols[i]])
            max_bin = max(df[feature_cols[i]])

            # print(min_bin, max_bin, n_bins, type(df[feature_cols[i]].values), is_numeric)
            bin_width = (max_bin - min_bin) / n_bins

            if hist_check_int and df[feature_cols[i]].astype(float).apply(float.is_integer).all():
                bin_array = np.arange(min_bin, max_bin + np.ceil(bin_width), np.ceil(bin_width))
            else:
                bin_array = np.arange(min_bin, max_bin + bin_width, bin_width)

            df.groupby(class_col)[feature_cols[i]].hist(alpha=0.4, ax=ax, density=True, bins=bin_array,
                                                      label=class_col)
            current_handles, current_labels = ax.get_legend_handles_labels()
            for idx, grp in enumerate(df.groupby(class_col).groups.keys()):
                current_labels[idx] = current_labels[idx] + "_" + str(grp)

            ax.set_title(title_prefix + feature_cols[i])
            ax.legend(current_handles, current_labels)
        plt.tight_layout()


# TODO perform random forest for important features
def plot_feature_importance_xgb(model_xgb, feature_names=None):
    if isinstance(model_xgb, xgb.XGBModel):
        importance = model_xgb.get_booster().get_fscore()
    else:
        importance = model_xgb.get_fscore()
    importance = sorted(importance.items(), key=operator.itemgetter(1))

    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()


    if (feature_names is not None) :
        if (len(feature_names) == len(df)):
            print("# features different between model & submission, assuming indexing...")
            print("# inputted:", len(feature_names), ", # model:", len(df))
        for f, feat_num in enumerate(df['feature']):
            print(feat_num)
            df.replace({'feature': {feat_num: feature_names[int(feat_num[1:])]}}, inplace=True)
    else:
        print(df["feature"])

    plt.figure()
    df.plot()
    df.plot(kind='barh', x='feature', y='fscore', legend=False)
    plt.title('Feature Importance')
    plt.xlabel('relative importance')
