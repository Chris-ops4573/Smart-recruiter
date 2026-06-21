from helper.profile_set import reason_embeddings, reason_names

import numpy as np

career_emb = np.load("../embeddings/career_embeddings.npy")

def relevant_field(index, skill_count):
    sims = np.dot(
        reason_embeddings,
        career_emb[index]
    )

    best_reason = reason_names[np.argmax(sims)]

    return f"Strong match due to experience building {best_reason} systems with {skill_count} aligned skills"