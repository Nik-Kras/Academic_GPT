import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import gaussian_kde
import json
import pickle
import paragraph_evaluation.training_dataset as td 
from paragraph_evaluation.utils import model_ask


def call_all_models(paragraph) -> dict:
    """ Stores output of each prompted model for a single paragraph """
    d = dict()
    for ind, prompt in enumerate(td.PROMPTS):
        model_result = model_ask(paragraph, prompt)
        d.update({f"model_{ind}": json.loads(model_result.content)})
    return d


def get_all_models_called_for_dataset() -> pd.DataFrame:
    """ Stores output of each prompted model for each paragraph from dataset """
    print("Calling GPT to evaluate datset...")
    print("Good paragraphs...")
    results = []
    for paragraph in tqdm(td.BAD_PARAGRAPHS):
        d = {"paragraph": paragraph, "type": "bad"}
        d.update(call_all_models(paragraph))
        results.append(d)

    print("Bad paragraphs...")
    for paragraph in tqdm(td.GOOD_PARAGRAPHS):
        d = {"paragraph": paragraph, "type": "good"}
        d.update(call_all_models(paragraph))
        results.append(d)
    
    df_results = pd.DataFrame(results)
    df_results.to_csv("paragraph_evaluation/data/DatasetAllPrompts.csv")
    print("Called all prompts for the whole dataset and saved thm to DatasetAllPrompts.csv")
    
    return df_results


def transform_dataset(df) -> pd.DataFrame:
    """ Creates a column MultiIndex DataFrame (model_x, feature_y) from raw dataset """

    # Remove unneccessary column with input text data, we analyse the model's output in regards to "good"/"bad" labels only.
    df = df.drop(['paragraph'], axis=1)

    # Now, we expand these dictionaries into sub-columns
    df_expanded = pd.json_normalize(df.drop(columns=['type']).to_dict('records'))

    # Create a new DataFrame with multi-level columns for better organization
    multi_col_df = pd.concat([df[['type']], df_expanded], axis=1)
    multi_col_df.columns = pd.MultiIndex.from_tuples([tuple(c.split('.')) if '.' in c else (c, '') for c in multi_col_df.columns])

    multi_col_df.to_csv("paragraph_evaluation/data/DatasetAllPrompts_multiindex.csv")
    print("Transformed the dataset to MultiIndex format and saved it as DatasetAllPrompts_multiindex.csv")
    return multi_col_df


def evaluate_model_fit_per_feature(df) -> pd.DataFrame:
    """ 
        For each single feature it calculates an overlap of distribution each model gives.
        The lower the value is, the less similar distributions are. (p-test) 
        When the value is 1 -> distributions are the same.
        Select the model with least p-value or create weights for models based on p-values
    """
    from scipy.stats import ranksums

    features = {t for t in df.columns.get_level_values(1) if t != ""}
    model_names = sorted(list(set([c for c in set(df.columns.get_level_values(0)) if c.startswith("model")])))

    # Split dataset to evaluation of good paragraphs and bad paragraphs
    bad_text = df[df["type"] == "bad"]
    good_text = df[df["type"] == "good"]

    evaluation = {}
    for feature in features:
        evaluation[feature] = []
        
        # Selects datapoints for one single feature from MultiIndex DataFrame columns: (model, feature)
        bad_model = bad_text.xs(feature, axis=1, level=1, drop_level=False)
        good_model = good_text.xs(feature, axis=1, level=1, drop_level=False)

        # Evaluate p-value for each model for the given feature
        for model in model_names:
            evaluation[feature].append(ranksums(bad_model[model], good_model[model]).pvalue[0])
            
    model_fit_df = pd.DataFrame.from_dict(evaluation, orient='index', columns=model_names)
    model_fit_df.to_csv("paragraph_evaluation/data/ModelFeatureFit.csv")
    print("Measured distribution similarity for bad and good paragraphs for each feature and model. Saved Model fittness for each feature in ModelFeatureFit.csv")
    return model_fit_df


def select_best_fit(model_fit) -> pd.Series:
    """ For ModelFeatureFit it finds models with least p-values, the best fit for given feature """
    res = {}
    for key, eval_list in model_fit.iterrows():
        res[key] = str(eval_list.idxmin())
    res = pd.Series(res)
    res.to_csv("paragraph_evaluation/data/FeatureModelBestFit.csv", header=False)
    print("Found the models that fit best for each feature separately and saved results in FeatureModelBestFit.csv")
    return res


def train_kde(dataset, feature_model_map):
    """ Saves KDE params for the dataset using given mapping feature <-> model """
    
    # Split dataset to evaluation of good paragraphs and bad paragraphs
    bad_text = dataset[dataset["type"] == "bad"]
    good_text = dataset[dataset["type"] == "good"]

    # Get KDE params for good and bad distributions
    res = {}
    for feature, model in feature_model_map.items():

        good_values = good_text[model][feature].to_list()
        bad_values = bad_text[model][feature].to_list()
        
        # To avoid error with numpy.linalg.LinAlgError: 1-th leading minor of the array is not positive definite -> adding small noise
        # Have to make sure that I don't have a list of same values
        good_values += np.random.normal(0,0.1,len(good_values))
        bad_values += np.random.normal(0,0.1,len(bad_values))

        res[feature] = {
            "good": gaussian_kde(good_values),
            "bad": gaussian_kde(bad_values)
        }
        
    f = open('paragraph_evaluation/data/kde_models.pckl', 'wb')
    pickle.dump(res, f)
    f.close()
    print("Stored KDE distribution parameters for each feature in kde_models.pckl")

    return res


def train_and_save_parameters():
    """ Implements training to find and save statistical parameters for further paragraph estimations """

    raw_dataset = get_all_models_called_for_dataset()
    dataset = transform_dataset(raw_dataset)
    model_fit_df = evaluate_model_fit_per_feature(dataset)
    feature_model_map = select_best_fit(model_fit_df)
    kde_params = train_kde(dataset, feature_model_map)


if __name__ == "__main__":
    train_and_save_parameters()
