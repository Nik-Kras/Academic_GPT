import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from paragraph_evaluation.utils import split_dataset_to_good_bad


def visualise_one_model_all_features_distribution(df, model_ind):
    """ Visualised one model of DatasetAllPrompts_multiindex dataset """

    model_name = f"model_{model_ind}"
    if model_name not in df.columns.levels[0]:
        print("Wrong model. Try one of: {}".format(df.columns.levels[0]))
    
    good_text, bad_text = split_dataset_to_good_bad(df)

    bad_model = bad_text[model_name]
    good_model = good_text[model_name]

    # Positions for Good and Bad
    positions_good = np.array(range(1, len(good_model.columns)*3, 3))
    positions_bad = positions_good + 1

    plt.figure(figsize=(10,6))

    # Boxplot for good_model
    bp_good = plt.boxplot([good_model[col] for col in good_model.columns], positions=positions_good, widths=0.35, labels=[f"{col}_good" for col in good_model.columns], patch_artist=True, boxprops=dict(facecolor='lightblue'))

    # Boxplot for DataFrame 4
    bp_bad = plt.boxplot([bad_model[col] for col in bad_model.columns], positions=positions_bad, widths=0.35, labels=[f"{col}_bad" for col in bad_model.columns], patch_artist=True, boxprops=dict(facecolor='lightgreen'))

    plt.title(f'Model #{model_ind}. Feature Distributions for Bad and Good paragraphs')
    plt.ylabel('Values')
    plt.xticks(rotation=45, ticks=np.arange(1.5, len(good_model.columns)*3, 3), labels=good_model.columns)
    plt.legend([bp_good["boxes"][0], bp_bad["boxes"][0]], ['Good', 'Bad'], loc='upper right')
    plt.grid(True)
    plt.show()


def visualise_all_models_distributions(df):
    """ Visualises distributions of all features for each model of DatasetAllPrompts_multiindex dataset """
    
    model_indexes = sorted(list(set([c[-1] for c in set(df.columns.get_level_values(0)) if c[-1].isdigit()])))
    for i in model_indexes:
        visualise_one_model_all_features_distribution(df, i)


def compare_models_on_feature(df, feature):
    """ Draws distributions each model produces for one specific feature """

    set_of_features = set(df.columns.get_level_values(1))
    if feature not in set_of_features:
        print("Wrong feature. Try one of {}".format(set_of_features))
        return None

    good_text, bad_text = split_dataset_to_good_bad(df)

    bad_model = bad_text.xs(feature, axis=1, level=1, drop_level=False)
    good_model = good_text.xs(feature, axis=1, level=1, drop_level=False)

    # Positions for Good and Bad
    positions_good = np.array(range(1, len(good_model.columns)*3, 3))
    positions_bad = positions_good + 1

    plt.figure(figsize=(10,6))

    # Boxplot for good_model
    bp_good = plt.boxplot([good_model[col] for col in good_model.columns], positions=positions_good, widths=0.35, labels=[f"{col}_good" for col in good_model.columns], patch_artist=True, boxprops=dict(facecolor='lightblue'))

    # Boxplot for DataFrame 4
    bp_bad = plt.boxplot([bad_model[col] for col in bad_model.columns], positions=positions_bad, widths=0.35, labels=[f"{col}_bad" for col in bad_model.columns], patch_artist=True, boxprops=dict(facecolor='lightgreen'))

    plt.title(f'Comparing models on feature: {feature}')
    plt.ylabel('Values')
    plt.xticks(rotation=45, ticks=np.arange(1.5, len(good_model.columns)*3, 3), labels=good_model.columns)
    plt.legend([bp_good["boxes"][0], bp_bad["boxes"][0]], ['Good', 'Bad'], loc='upper right')
    plt.grid(True)
    plt.show()
    
def compare_all_features_to_all_models(df):
    """ For each feature it draws box-plots to compare distributions each model gives """
    features = {t for t in df.columns.get_level_values(1) if t != "" and "Unnamed" not in t}
    for feature in features:
        compare_models_on_feature(df, feature)
        
        
def visualise_one_kde(good_values, bad_values, title):

    from scipy.stats import gaussian_kde

    # Fit KDE to data
    good_kde = gaussian_kde(good_values)
    bad_kde = gaussian_kde(bad_values)

    # Values over which we'll calculate the KDE
    x_min = min(min(good_values), min(bad_values)) - 10
    x_max = max(max(good_values), max(bad_values)) + 10
    x_values = np.linspace(x_min, x_max, 1000)

    # Evaluate the KDE here
    good_kde_values = good_kde(x_values)
    bad_kde_values = bad_kde(x_values)

    # Plotting the KDEs
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.plot(x_values, good_kde_values, label='Good KDE', color='blue')
    plt.plot(x_values, bad_kde_values, label='Bad KDE', color='red')
    plt.scatter(good_values, np.zeros_like(good_values), color='blue', zorder=5, label='Good Data Points')
    plt.scatter(bad_values, np.zeros_like(bad_values), color='red', zorder=5, label='Bad Data Points')
    plt.legend()
    plt.grid(True)
    plt.show()


def visualise_all_kde(df, df_model_feature_fit):
    """ Draws KDEs for each feature usinng the models that describe the feature the best """

    good_text, bad_text = split_dataset_to_good_bad(df)

    for feature, model in df_model_feature_fit.items():
        good_values = good_text[model][feature].to_list()
        bad_values = bad_text[model][feature].to_list()
        title = "{} for feature {}".format(model, feature)
        visualise_one_kde(good_values, bad_values, title)
