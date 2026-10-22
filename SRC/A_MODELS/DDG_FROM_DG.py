""""
Copyright (C) 2025 ETH Zurich, Stefan J.P. Chrin, Gregory Landrum, Sereina Riniker
"""

import itertools, json

def ddG_from_dG(input_json, output_json):
    """
    input:  json file path containing list of dicts [ {target:{ligand:{prediction,experimental},...}}, ...]
    output: json file path containing same structure but ddG values per ligand pair
    """
    # load data
    with open(input_json, "r") as f:
        all_splits = json.load(f)

    ddG_all_splits = []  # list for all splits

    # iterate splits
    for split in all_splits:
        split_ddG = {}
        for target, ligs in split.items():
            lig_names = sorted(ligs.keys())  # sorted alphabetical order
            pairs = list(itertools.combinations(lig_names, 2))
            target_ddG = {}
            # iterate pairs
            for a, b in pairs:
                pred_a = ligs[a]["prediction"]
                pred_b = ligs[b]["prediction"]
                exp_a = ligs[a]["experimental"]
                exp_b = ligs[b]["experimental"]
                # compute ddG
                ddG_pred = pred_b - pred_a
                ddG_exp = exp_b - exp_a
                pair_name = f"{a}__{b}"
                # store
                target_ddG[pair_name] = {
                    "prediction": ddG_pred,
                    "experimental": ddG_exp,
                }
            split_ddG[target] = target_ddG
        ddG_all_splits.append(split_ddG)

    # save
    with open(output_json, "w") as f:
        json.dump(ddG_all_splits, f, indent=2)

    print("ddG values saved :)  ->", output_json)
