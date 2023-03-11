import pymc as pm
import numpy as np
from glob import glob
from toolz import pipe, partial
import pandas as pd
from os.path import join, splitext
from sklearn.preprocessing import LabelEncoder


def get_data(sackmann_dir, tour="atp", keep_davis_cup=False, discard_retirements=True):

    all_csvs = glob(join(sackmann_dir, f"*{tour}_matches_????.csv"))
    all_csvs = sorted(all_csvs, key=lambda x: int(splitext(x)[0][-4:]))

    levels_to_drop = ["C", "S"]

    if not keep_davis_cup:
        levels_to_drop.append("D")

    data = pipe(
        all_csvs,
        # Read CSV
        lambda y: map(partial(pd.read_csv, encoding="ISO=8859-1"), y),
        # Drop NAs in important fields
        lambda y: map(
            lambda x: x.dropna(subset=["winner_name", "loser_name", "score"]), y
        ),
        # Drop retirements and walkovers
        # TODO: Make this optional
        lambda y: map(
            lambda x: x
            if not discard_retirements
            else x[~x["score"].astype(str).str.contains("RET|W/O|DEF|nbsp|Def.")],
            y,
        ),
        # Drop scores that appear truncated
        lambda y: map(lambda x: x[x["score"].astype(str).str.len() > 4], y),
        # Drop challengers and futures
        # TODO: Make this optional too
        lambda y: map(lambda x: x[~x["tourney_level"].isin(levels_to_drop)], y),
        pd.concat,
    )

    round_numbers = {
        "R128": 1,
        "RR": 1,
        "R64": 2,
        "R32": 3,
        "R16": 4,
        "QF": 5,
        "SF": 6,
        "F": 7,
    }

    # Drop rounds outside this list
    to_keep = data["round"].isin(round_numbers)
    data = data[to_keep]

    # Add a numerical round number
    data["round_number"] = data["round"].replace(round_numbers)

    # Add date information
    data["tourney_date"] = pd.to_datetime(
        data["tourney_date"].astype(int).astype(str), format="%Y%m%d"
    )
    data["year"] = data["tourney_date"].dt.year

    # Sort by date and round and reset index
    data = data.sort_values(["tourney_date", "round_number"])
    data = data.reset_index(drop=True)

    data["pts_won_serve_winner"] = data["w_1stWon"] + data["w_2ndWon"]
    data["pts_won_serve_loser"] = data["l_1stWon"] + data["l_2ndWon"]

    data["pts_played_serve_winner"] = data["w_svpt"]
    data["pts_played_serve_loser"] = data["l_svpt"]

    # Add serve % won
    data["spw_winner"] = (data["w_1stWon"] + data["w_2ndWon"]) / data["w_svpt"]
    data["spw_loser"] = (data["l_1stWon"] + data["l_2ndWon"]) / data["l_svpt"]

    data["spw_margin"] = data["spw_winner"] - data["spw_loser"]

    return data


def compute_game_margins(string_scores):
    def compute_margin(sample_set):

        if "[" in sample_set:
            return 0

        try:
            split_set = sample_set.split("-")
            margin = int(split_set[0]) - int(split_set[1].split("(")[0])
        except ValueError:
            margin = np.nan

        return margin

    margins = pipe(
        string_scores,
        lambda y: map(lambda x: x.split(" "), y),
        lambda y: map(lambda x: [compute_margin(z) for z in x], y),
        lambda y: map(sum, y),
        partial(np.fromiter, dtype=np.float),
    )

    return margins


def get_player_info(sackmann_dir, tour="atp"):

    player_info = pd.read_csv(
        join(sackmann_dir, f"{tour}_players.csv"),
        header=None,
        names=[
            "ID",
            "First Name",
            "Last Name",
            "Handedness",
            "Birthdate",
            "Nationality",
        ],
    )

    player_info = player_info.dropna()

    str_date = player_info["Birthdate"].astype(int).astype(str)
    str_date = pd.to_datetime(str_date, format="%Y%m%d")

    player_info["Birthdate"] = str_date

    return player_info


def fetch_tennis_model(start_year, sackmann_dir="./examples/tennis_atp"):

    # TODO: Maybe download this if needed
    df = get_data(sackmann_dir)

    rel_df = df[df["tourney_date"].dt.year >= start_year]

    encoder = LabelEncoder()

    encoder.fit(
        rel_df["winner_name"].values.tolist() + rel_df["loser_name"].values.tolist()
    )

    names = encoder.classes_
    winner_ids = encoder.transform(rel_df["winner_name"])
    loser_ids = encoder.transform(rel_df["loser_name"])
    n_players = len(names)

    with pm.Model() as hierarchical_model:
        # Hyperpriors for group nodes
        sigma_player = pm.HalfNormal("sigma_player", 1.0)

        player_skills = pm.Normal(
            "player_skills", mu=0.0, sigma=sigma_player, shape=n_players
        )

        logit_skills = player_skills[winner_ids] - player_skills[loser_ids]

        # Data likelihood
        lik = pm.Bernoulli(
            "win_lik", logit_p=logit_skills, observed=np.ones(winner_ids.shape[0])
        )

    return {"model": hierarchical_model, "encoder": encoder}


def fetch_correlated_tennis_model(start_year, sackmann_dir="./examples/tennis_atp"):

    # TODO: Duplicated with the other tennis data. Maybe put some of this into a function.
    from sklearn.preprocessing import LabelEncoder

    df = get_data(sackmann_dir)

    rel_df = df[df["tourney_date"].dt.year >= start_year]

    encoder = LabelEncoder()

    encoder.fit(
        rel_df["winner_name"].values.tolist() + rel_df["loser_name"].values.tolist()
    )

    names = encoder.classes_
    winner_ids = encoder.transform(rel_df["winner_name"])
    loser_ids = encoder.transform(rel_df["loser_name"])

    n_players = len(names)

    surface_encoder = LabelEncoder()

    surface_encoder.fit(rel_df["surface"])
    surf_names = surface_encoder.classes_
    n_surf = len(surf_names)

    surface_ids = surface_encoder.transform(rel_df["surface"])

    with pm.Model() as model:
        chol, corr, stds = pm.LKJCholeskyCov(
            "chol",
            n=len(surf_names),
            eta=2.0,
            sd_dist=pm.Exponential.dist(1.0),
            compute_corr=True,
        )

        player_skills = pm.MvNormal(
            "player_skills", np.zeros(n_surf), chol=chol, shape=(n_players, n_surf)
        )

        logit_skills = (
            player_skills[winner_ids, surface_ids]
            - player_skills[loser_ids, surface_ids]
        )

        lik = pm.Bernoulli(
            "win_lik", logit_p=logit_skills, observed=np.ones(winner_ids.shape[0])
        )

    return model
