import numpy as np
from bambi import Model
import pymc3 as pm
import json
import os
import pandas as pd


def load_arm_data(
    subdir,
    model_name,
    json_dir="/home/martin/projects/papers/lrvb/SharedExampleModels/ARM_json/",
):

    chapter = subdir.split("/")[-1]

    filename = os.path.join(json_dir, chapter, model_name + ".data.json")

    return json.load(open(filename))


def create_bambi_model(data_df, family_name, var_names, fixed_str, groups):

    model = Model(data_df)
    model.fit(fixed_str, group_specific=groups, run=False, family=family_name)
    model.build()

    return model.backend.model


def scale_vector(vector):

    return (vector - vector.mean()) / vector.std()


def load_arm_model(model_name, arm_formula_df, json_data_dir, scale_data=False):
    """
    Loads an ARM model.
    Args:
        model_name: The model to load [e.g. "radon_inter_vary"].
        arm_formula_df: The DataFrame containing the model formulae.
        json_data_dir: The path containing the json files with data.
    Returns:
    A PyMC model object encoding the model.
    """

    arm_formula_df = arm_formula_df.set_index("model_name", drop=False)
    arm_formula_df["to_scale_str"] = (
        arm_formula_df["to_scale_str"].fillna("").astype(str)
    )

    cur_row = arm_formula_df.loc[model_name]
    fixed_formula_str = cur_row["python_fixed_str"]

    data = load_arm_data(cur_row["subdir"], cur_row["model_name"], json_data_dir)

    if cur_row["python_library"] == "bambi":

        family = cur_row["bambi_family"]
        var_names = cur_row["bambi_var_names_str"].split(" ")
        groups = cur_row["bambi_groups_str"].split(";")
        rel_data = {x: np.array(y) for x, y in data.items() if x in var_names}

        if scale_data:

            cont_vars = cur_row["to_scale_str"].split(" ")

            rel_data = {
                x: scale_vector(y) if x in cont_vars else y for x, y in rel_data.items()
            }

        df = pd.DataFrame(rel_data)
        return create_bambi_model(df, family, var_names, fixed_formula_str, groups)

    else:

        assert cur_row["python_library"] == "pymc3_glm"
        data = {x: np.array(y) for x, y in data.items()}

        if scale_data:

            cont_vars = cur_row["to_scale_str"].split(" ")

            data = {
                x: scale_vector(y) if x in cont_vars else y for x, y in data.items()
            }

        with pm.Model() as m:

            pm.glm.linear.GLM.from_formula(
                fixed_formula_str,
                data,
                family=cur_row["pymc3_family"],
            )

        return m


def load_all_arm_models(arm_formula_df, json_data_dir, scale_data=False):
    """
    Loads all ARM models.
    Args:
        arm_formula_df: The DataFrame containing the model formulae.
        json_data_dir: The path containing the json files with data.
    Returns:
    A dictionary from model_name to PyMC objects.
    """

    model_names = arm_formula_df["model_name"].values

    results = dict()

    for cur_model_name in model_names:

        try:
            results[
                cur_model_name
            ] = lambda cur_model_name=cur_model_name, arm_formula_df=arm_formula_df, json_data_dir=json_data_dir: load_arm_model(
                cur_model_name, arm_formula_df, json_data_dir, scale_data=scale_data
            )
        except Exception as e:
            print(f"Failed to load model {cur_model_name}. Error was:")
            print(e)

    return results
