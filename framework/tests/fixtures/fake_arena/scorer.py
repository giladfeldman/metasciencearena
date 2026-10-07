def score(player_output, ground_truth):
    return {"primary": 1.0 if player_output.get("label") == ground_truth.get("label") else 0.0, "breakdown": {}}
