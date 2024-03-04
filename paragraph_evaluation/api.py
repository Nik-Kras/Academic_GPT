import pandas as pd
import numpy as np
import json
import paragraph_evaluation.training_dataset as td
from paragraph_evaluation.parameters import FEATURE_MODEL_MAP, STATISTICAL_MODEL
from paragraph_evaluation.utils import model_ask
from paragraph_evaluation.binary_classifier import predict


def get_metrics_from_mandatory_models(paragraph, logs: bool = True) -> dict:
    """
    <PARAGRAPH> ->
    {"model_1": ..., "model_3": ...}
    """
    only_mandatory_models = set(FEATURE_MODEL_MAP)
    model_index = [int(c[-1]) for c in only_mandatory_models]

    model_output = {}
    for ind in model_index:
        model_output[f"model_{ind}"] = json.loads(model_ask(paragraph, td.PROMPTS[ind]).content)

    if logs:
        print("General evaluation by every model")
        print(pd.DataFrame(model_output))
    return model_output


def select_by_best_model_per_feature(all_models_metrics, logs: bool = True) -> dict:
    """
    {"model_1": {"feature_1": ..., ..., "feature_N": ...}, "model_3": ...} ->
    {"feature_1": ..., "feature_2": ...}
    """
    output = {}
    for feature, model in FEATURE_MODEL_MAP.items():
        output[feature] = all_models_metrics[model][feature]

    if logs:
        print("Left only values from most useful models")
        print(pd.Series(output))
    return output


def get_feature_probabilities(feature_metrics) -> dict:
    """ Converting absolute values to probabilites according to trained statistical KDE distribution parameters
    {"feaure_1": 94, ...} - > {"feaure_1": 85%, ...}
    """
    output = {}
    for feature, kdes in STATISTICAL_MODEL.items():

        # Calculate the probability density of the new data point in each distribution
        good_prob_density = kdes["good"].evaluate(feature_metrics[feature])[0]
        bad_prob_density = kdes["bad"].evaluate(feature_metrics[feature])[0]

        # Normalize the probabilities so they sum to 1
        total_density = good_prob_density + bad_prob_density
        prob_good = good_prob_density / total_density if total_density > 0 else 0
        # prob_bad = bad_prob_density / total_density if total_density > 0 else 0

        output[feature] = prob_good
        
    return output


def view_prediction_output(feature_metrics, feature_probabilities, label_prediction):
    """ Prints an output with hight interpretability """
    
    out = {}
    for key, value in feature_probabilities.items():
        out[key] = "{} - {:.1f}%".format(feature_metrics[key], 100*value)

    print("How good each criteria is:")
    print(pd.Series(out))

    label, prob = list(label_prediction.items())[0]
    print("Overall Label prediction: {} - {:.1f}%".format(label, 100*prob))


def evaluate_paragraph(paragraph: str, logs: bool = True) -> dict:
    """ Returns an evaluation of paragraph by criterias aka features """
    all_models_metrics = get_metrics_from_mandatory_models(paragraph, logs)
    feature_metrics = select_by_best_model_per_feature(all_models_metrics, logs)
    feature_probabilities = get_feature_probabilities(feature_metrics)
    label_prediction = predict(pd.Series(feature_metrics))
    
    if logs:
        view_prediction_output(feature_metrics, feature_probabilities, label_prediction)
    
    all_probs = feature_probabilities.copy()
    all_probs.update({"classification": label_prediction})
    return all_probs


if __name__ == "__main__":
    paragraph = """Abstract: This paper examines popular model ToMnet developed by DeepMind and influenced the field of study of Machine Theory of Mind. While original ToMnet implementation is closed and its alternative ToMnet+ was developed in no longer actual framework - TensorFlow 1, the new model was proposed, developed and used for experiments: ToMnet-N (Theory of Mind Network by Nikita). This model solves a trajectory prediction problem for an observed player in the game in auto-regressive manner. The implementation is done in more modern TensorFlow 2. However, the most significant theoretical impact the paper makes is it argues with original ToMnet research. I claim that either ToMnet or any other network from the family: ToMnet+, ToM2c, Trait-ToM, ToMnet-G and others never achieved Theory of Mind and never will, as the implementation only considers pattern recognition which is only merely similar to Theory-Theory approach in Theory of Mind. I came up with this conclusion after repeating most of original experiments from ToMnet paper using my ToMnet-N model and my custom game and A* based bot player."""
    evaluate_paragraph(paragraph)
