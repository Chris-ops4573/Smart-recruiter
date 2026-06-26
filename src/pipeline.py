from helper.score import score_embeddings, score_skill_keyword, score_behavior, honeypot_penalty
from helper.reasons import relevant_field

import json
import csv
import sys

it = 0

scores = []
with open("../data/candidates.jsonl") as f:
    for line in f:
        candidate = json.loads(line)

        career_score, profile_score = score_embeddings(it)
        skill_score, skill_count, matched_skills = score_skill_keyword(candidate)
        behaviour_score = score_behavior(candidate)

        honey_penalty = honeypot_penalty(candidate, skill_score, career_score)

        final_score = (
            0.70 * career_score +
            0.20 * behaviour_score +
            0.10 * profile_score + 
            0.10 * skill_score - 
            honey_penalty
        )
        scores.append((
            candidate["candidate_id"], 
            candidate,
            final_score, 
            skill_count,
            matched_skills,
            it,
            career_score, 
            profile_score, 
            skill_score, 
            behaviour_score, 
            honey_penalty
        ))

        it += 1

scores.sort(key = lambda x:x[2], reverse = True)
top_100 = scores[:100]

with open(sys.argv[1], "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "candidate_id",
        "rank",
        "score",
        "reasoning"
    ])

    rank = 1
    for candidate_id, candidate, score, skill_count, matched_skills, index, career_score, _, _, behaviour_score, _ in top_100:
        reasoning = relevant_field(candidate, index, skill_count, matched_skills, career_score, behaviour_score)

        writer.writerow([
            candidate_id, rank, score, reasoning
        ])
        rank += 1