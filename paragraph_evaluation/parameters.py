import pandas as pd
import pickle
import os

# Absolute path to file locations, to avoid visibility issues
current_dir = os.path.dirname(os.path.abspath(__file__))

# Use the constructed path to build the absolute path to the data files
feature_model_map_path = os.path.join(current_dir, "data", "FeatureModelBestFit.csv")
kde_models_path = os.path.join(current_dir, "data", "kde_models.pckl")
dataset_multiidenx_path = os.path.join(current_dir, "data", "DatasetAllPrompts_multiindex.csv")

# Which model gives the most meaningful values for given feature (meaningful = easy to rely on to destinguish between good and bad articles)
FEATURE_MODEL_MAP = pd.read_csv(feature_model_map_path, index_col=0, header=None).squeeze("columns")

# Statistical params for probability estimation
f = open(kde_models_path, 'rb')
STATISTICAL_MODEL = pickle.load(f)
f.close()

DATASET_MULTIIDEX = pd.read_csv(dataset_multiidenx_path, header=[0, 1], index_col=0)
