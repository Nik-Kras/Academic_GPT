from paragraph_evaluation.parameters import FEATURE_MODEL_MAP
from openai import OpenAI
import os
import pandas as pd

API_KEY = os.environ['API_KEY']

def model_ask(paragraph, prompt):

    client = OpenAI(api_key=API_KEY)

    completion = client.chat.completions.create(
      model="gpt-3.5-turbo-0125",
      messages=[
        {"role": "system", "content": prompt},
        {"role": "user", "content": paragraph}
      ],
      response_format={ "type": "json_object" }
    )

    return completion.choices[0].message


def split_dataset_to_good_bad(df):
    bad_index = df["type"] == "bad"
    bad_index = bad_index.squeeze()
    good_index = df["type"] == "good"
    good_index = good_index.squeeze()
    bad_text = df[bad_index]
    good_text = df[good_index]

    return good_text, bad_text


def replace_multiindex_with_best_model_feature_fit(df):
    """ Give it MultiIndex dataset and get a dataset with only one value per feature that fits the best """
    res = []
    features = [x for x in set(df.columns.get_level_values(1)) if "Unnamed" not in x]
    for index, row in df.iterrows():
        
        d = {"type": row["type"].squeeze()}
        for feature in features:
            d.update({f"{feature}": row[FEATURE_MODEL_MAP[feature]][feature]})
            
        res.append(d)
        
    df_res = pd.DataFrame(res)
    df_res = df_res.reindex(sorted(df_res.columns), axis=1) # Sort columns alphabetically
    return df_res
