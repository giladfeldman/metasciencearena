def score(player_output, ground_truth):
    if player_output.get("label") == ground_truth.get("label"):
        return {"primary": 1.0, "breakdown": {}, "findings": []}
    return {
        "primary": 0.0,
        "breakdown": {},
        "findings": [{
            "category": "wrong_label",
            "evidence": player_output.get("label"),
            "correct_value": ground_truth.get("label"),
        }],
    }
