from helper.score import score_embeddings, score_skill_keyword, score_behavior, honeypot_penalty
from helper.reasons import relevant_field

import numpy as np
import json
import csv
import sys
import os

def rank_candidates(candidate_count, candidate_file, file_path, embeddings_dir):
    it = 0
    career_emb = np.load(os.path.join(embeddings_dir, "career.npy"))
    profile_emb = np.load(os.path.join(embeddings_dir, "profile.npy"))

    career_jd_emb = np.load(os.path.join(embeddings_dir, "jd_career.npy"))
    profile_jd_emb = np.load(os.path.join(embeddings_dir, "jd_profile.npy"))
    reason_emb = np.load(os.path.join(embeddings_dir, "reason.npy"))

    scores = []
    with open(candidate_file) as f:
        for line in f:
            candidate = json.loads(line)

            career_score, profile_score = score_embeddings(career_emb[it], profile_emb[it], career_jd_emb, profile_jd_emb)
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
    top = scores[:candidate_count] 

    if it + 1 < candidate_count:
        print("Error: Ranked count exceeds total candidates in file")
        return

    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "candidate_id",
            "rank",
            "score",
            "reasoning"
        ])

        rank = 1
        for candidate_id, candidate, score, skill_count, matched_skills, index, career_score, _, _, behaviour_score, _ in top:
            reasoning = relevant_field(candidate, index, skill_count, matched_skills, career_score, behaviour_score, career_emb[index], reason_emb)

            writer.writerow([
                candidate_id, rank, score, reasoning
            ])
            rank += 1

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python syc/pipeline.py total_rank data_file ranked_file embedding_dir")
        exit()

    rank_candidates(int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4])