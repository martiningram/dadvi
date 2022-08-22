import numpy as np
import bambi as bmb
import pymc as pm
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


def add_bambi_family(arm_df):

    arm_df["bambi_family"] = arm_df["family"].replace(
        {"gaussian()": "gaussian", 'binomial(link="logit")': "bernoulli"}
    )

    return arm_df


def load_arm_model(arm_df_row, json_data_dir):

    cur_row = arm_df_row

    data = load_arm_data(cur_row["subdir"], cur_row["model_name"], json_data_dir)

    family = cur_row["bambi_family"]
    formula = cur_row["formula_str"].replace("log(", "np.log(")
    formula = formula.replace("as.factor(", "C(")
    formula = formula.replace("factor(", "C(")

    # Create dataset -- all of the relevant vectors should have N entries
    N = data["N"][0]
    rel_data = {x: y for x, y in data.items() if len(y) == N}

    rel_df = pd.DataFrame(rel_data)

    model = bmb.Model(formula, rel_df, family=family)

    model.build()
    backend_model = model.backend

    return {"bambi_model": model, "pymc_model": backend_model.model}
