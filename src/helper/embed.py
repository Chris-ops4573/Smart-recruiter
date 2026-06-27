# Generate embeddings and store to disk
from sentence_transformers import SentenceTransformer

import numpy as np
import json
import os

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

#Embeddings to be stored to disk
career_embeddings = []
profile_embeddings = []

#For processing texts in batches
career_texts = []
profile_texts = []

count = 1

def embed(file_path, embedding_path):
    global count
    
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