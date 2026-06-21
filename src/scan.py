import json

with open("../data/candidates.jsonl") as f:
    for line in f:
        candidate = json.loads(line)

        if candidate["candidate_id"] == "CAND_0046525":
            print(json.dumps(candidate, indent=4))
            break