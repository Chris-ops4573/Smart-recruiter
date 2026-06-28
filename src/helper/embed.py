# Generate embeddings and store to disk
from sentence_transformers import SentenceTransformer
from profile_set import jd_career_text, jd_profile_text, reason_templates

import numpy as np
import json
import os
import sys

model = SentenceTransformer("BAAI/bge-base-en-v1.5")

def build_work_text(candidate):
    jobs = []

    for job in candidate["career_history"]:
        jobs.append(
            f"Role: {job['title']}\n"
            f"Company: {job['company']}\n"
            f"Industry: {job['industry']}\n"

            f"Responsilities:\n"
            f"Description: {job['description']}\n"
        )
    
    return "\n".join(jobs)

def build_profile_text(candidate):
    candidate_profile = candidate['profile']
    skills = ", ".join(
        skill["name"] for skill in candidate["skills"]
    )

    return (
        f"Current Role: {candidate_profile['current_title']}\n"
        f"Headline: {candidate_profile['headline']}\n"

        f"Professional summary:\n"
        f"Summary: {candidate_profile['summary']}\n"
        f"Skills: {skills}\n"
    )

def embed(file_path, embedding_path):
    #Embeddings to be stored to disk
    career_embeddings = []
    profile_embeddings = []

    #For processing texts in batches
    career_texts = []
    profile_texts = []

    count = 1

    with open(file_path) as f:
        for line in f:
            candidate = json.loads(line)
            career_text = build_work_text(candidate)
            profile_text = build_profile_text(candidate)

            career_texts.append(career_text)
            profile_texts.append(profile_text)
            
            if len(career_texts) == 256:
                print(count)
                count += 1

                career_embs = model.encode(
                    career_texts,
                    batch_size = 256,
                    normalize_embeddings = True
                )
                profile_embs = model.encode(
                    profile_texts,
                    batch_size = 256,
                    normalize_embeddings = True
                )

                career_embeddings.extend(career_embs)
                profile_embeddings.extend(profile_embs)

                career_texts.clear()
                profile_texts.clear()

        if career_texts:
            career_embs = model.encode(
                career_texts,
                batch_size = 256,
                normalize_embeddings = True
            )
            profile_embs = model.encode(
                profile_texts,
                batch_size = 256,
                normalize_embeddings = True
            )

            career_embeddings.extend(career_embs)
            profile_embeddings.extend(profile_embs)

            career_texts.clear()
            profile_texts.clear()

    #Save to disk
    np.save(
        os.path.join(embedding_path, "career.npy"),
        np.array(career_embeddings)
    )

    np.save(
        os.path.join(embedding_path, "profile.npy"),
        np.array(profile_embeddings)
    )

    # JD embeddings
    jd_career_embedding = model.encode(
        jd_career_text,
        normalize_embeddings=True
    )

    jd_profile_embedding = model.encode(
        jd_profile_text,
        normalize_embeddings=True
    )

    np.save(
        os.path.join(embedding_path, "jd_career.npy"),
        jd_career_embedding
    )

    np.save(
        os.path.join(embedding_path, "jd_profile.npy"),
        jd_profile_embedding
    )

    reason_embeddings = model.encode(
        list(reason_templates.values()),
        normalize_embeddings=True
    )

    np.save(
        os.path.join(embedding_path, "reason.npy"),
        reason_embeddings
    )

if __name__ == "__main__":
    if(len(sys.argv) != 3):
        print("Usage: python srcembed.py data_file embedding_dir")
        exit()

    embed(sys.argv[1], sys.argv[2])