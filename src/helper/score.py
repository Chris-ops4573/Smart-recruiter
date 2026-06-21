from helper.profile_set import required_skills, preffered_skills, jd_career_embedding, jd_profile_embedding

import json
import re
import numpy
from datetime import datetime
import numpy as np

profile_emb = np.load("../embeddings/profile_embeddings.npy")
career_emb = np.load("../embeddings/career_embeddings.npy")

#Helpers
def date_to_day(date):
    last_active_date = datetime.strptime(
        date,
        "%Y-%m-%d"
    )

    days_since_active = (
        datetime.now() - last_active_date
    ).days
    return days_since_active

def days_btw_dates(date1, date2):
    day1 = date_to_day(date1)
    day2 = date_to_day(date2)
    
    return day1 - day2

def normalize_skill(skill):
    skill = skill.lower().strip()

    skill = skill.replace("-", " ")
    skill = skill.replace("_", " ")

    skill = re.sub(r"\s+", " ", skill)

    return skill

#Scoring functions
def score_skill_keyword(candidate):
    required_skill = 0
    preffered_skill = 0

    skill_count = 0
    matched_skills = []

    for skill in candidate['skills']:
        skill_name = normalize_skill(skill["name"])

        if skill_name in required_skills:
            matched_skills.append(skill_name)
            skill_count += 1

            if skill['proficiency'] == "expert": 
                required_skill += 1
            elif skill['proficiency'] == "advanced":
                required_skill += 0.8
            elif skill['proficiency'] == "intermediate":
                required_skill += 0.5
            else:
                required_skill += 0.3

        elif skill_name in preffered_skills:
            matched_skills.append(skill_name)
            skill_count += 1

            if skill['proficiency'] == "expert": 
                preffered_skill += 1
            elif skill['proficiency'] == "advanced":
                preffered_skill += 0.8
            elif skill['proficiency'] == "intermediate":
                preffered_skill += 0.5
            else:
                preffered_skill += 0.3
    
    required_skill /= len(required_skills)
    preffered_skill /= len(preffered_skills)

    return 0.8 * required_skill + 0.2 * preffered_skill, skill_count, matched_skills

def score_embeddings(candidate_id):
    career_score = np.dot(
        jd_career_embedding,
        career_emb[candidate_id]
    )

    profile_score = np.dot(
        jd_profile_embedding,
        profile_emb[candidate_id]
    )

    return career_score, profile_score

def score_behavior(candidate):
    redrob_signals = candidate["redrob_signals"]

    open_work_score = 1 if redrob_signals["open_to_work_flag"] else 0
    response_score = redrob_signals["recruiter_response_rate"]
    notice_score = max(0, 1 - (redrob_signals["notice_period_days"]/120))

    days_since_active = date_to_day(redrob_signals["last_active_date"])
    recency_score = max(0, 1 - days_since_active/180)

    interview_score = redrob_signals["interview_completion_rate"]
    saved_score = min(1, redrob_signals["saved_by_recruiters_30d"]/10)
    github_activity_score = 0 if redrob_signals["github_activity_score"] <= 0 else redrob_signals["github_activity_score"]/100.0

    behavior_score = (
        0.30 * open_work_score +
        0.25 * response_score +
        0.15 * recency_score +
        0.10 * notice_score +
        0.10 * interview_score +
        0.05 * saved_score +
        0.05 * github_activity_score
    )
    return behavior_score

def honeypot_penalty(candidate, skill_score, career_score):
    penalty = 0

    total_days = 0
    for job in candidate["career_history"]:
        start_date = job["start_date"] 
        end_date = job["end_date"] if job["end_date"] else "2026-06-21"

        total_days += days_btw_dates(start_date, end_date)
    
    claimed_days = candidate["profile"]["years_of_experience"] * 365

    if(1.5 * total_days < claimed_days):
        penalty += 0.2
    
    if skill_score > 0.5 and career_score < 0.3:
        penalty += 0.4

    return penalty