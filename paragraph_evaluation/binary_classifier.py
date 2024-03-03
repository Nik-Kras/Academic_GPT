from paragraph_evaluation.parameters import DATASET_MULTIIDEX
from paragraph_evaluation.utils import split_dataset_to_good_bad, replace_multiindex_with_best_model_feature_fit
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import pandas as pd


MODEL_FILENAME = "data/binary_classifier.pickle"


def create_dataset():
    """ Creates a dataset splitted to train and test sets """
    
    dataset_with_best_features = replace_multiindex_with_best_model_feature_fit(DATASET_MULTIIDEX)
    
    good_text, bad_text = split_dataset_to_good_bad(dataset_with_best_features)
    
    good_train, good_test = train_test_split(good_text, test_size=0.2, random_state=42, shuffle=True)
    bad_train, bad_test = train_test_split(bad_text, test_size=0.2, random_state=42, shuffle=True)
    
    train = pd.concat([good_train, bad_train])
    test = pd.concat([good_test, bad_test])
    
    train_y, train_x = train["type"], train.drop(["type"], axis=1)
    test_y, test_x = test["type"], test.drop(["type"], axis=1)

    return train_y, train_x, test_y, test_x
    

def train_classifier(train_y, train_x):
    """ Train Logistic Regression classifier """
    model = LogisticRegression()
    model.fit(train_x, train_y)
    save_model(model)
    return model
    
    
def evaluate_classifier(train_y, train_x, test_y, test_x):
    """ Evaluates the classification model """
    model = load_model()
    train_predict = model.predict(train_x)
    test_predict = model.predict(test_x)
    test_accuracy = accuracy_score(test_y, test_predict)
    train_accuracy = accuracy_score(train_y, train_predict)
    
    print(f"Train Accuracy: {train_accuracy:.2f}")
    print(f"Test Accuracy: {test_accuracy:.2f}")
    

def save_model(model):
    pickle.dump(model, open(MODEL_FILENAME, "wb"))
    
    
def load_model():
    return pickle.load(open(MODEL_FILENAME, "rb"))
    
    
def predict(paragraph_metrics) -> float:
    """ Returns probability of the paragraph to be good based on vectory of metrics """
    model = load_model()
    input_data = paragraph_metrics.to_frame().T # Convert pd.Series to one-row DataFrame
    input_data = input_data.reindex(sorted(input_data.columns), axis=1) # Sort columns alphabetically
    probs = model.predict_proba(input_data)[0]
    
    classes = model.classes_
    # print(f"{classes[0]}:{100*probs[0]:.1f}% and {classes[1]}:{100*probs[1]:.1f}%")
    if probs[0] > 0.5:
        return {classes[0]: probs[0]}
    else:
        return {classes[1]: probs[1]}


if __name__ == "__main__":
    train_y, train_x, test_y, test_x = create_dataset()
    model = train_classifier(train_y, train_x)
    
    evaluate_classifier(train_y, train_x, test_y, test_x)
    
    for index, x in test_x.iterrows():
        print("Actual Label: {}".format(test_y.loc[index]))
        print("Predicted label: {}".format(predict(x)))
