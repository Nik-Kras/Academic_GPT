import pandas as pd
import pickle

# Which model gives the most meaningful values for given feature (meaningful = easy to rely on to destinguish between good and bad articles)
FEATURE_MODEL_MAP = pd.read_csv("paragraph_evaluation/data/feature_model_map.csv", index_col=0, header=None).squeeze("columns")

# Statistical params for probability estimation
f = open('paragraph_evaluation/data/kde_models.pckl', 'rb')
STATISTICAL_MODEL = pickle.load(f)
f.close()

DATASET_MULTIIDEX = pd.read_csv("paragraph_evaluation/data/DatasetAllPrompts_multiindex.csv", header=[0, 1], index_col=0)