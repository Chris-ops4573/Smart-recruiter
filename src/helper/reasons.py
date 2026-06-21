from helper.profile_set import reason_embeddings, reason_names
import numpy as np

career_emb = np.load("../embeddings/career_embeddings.npy")

def relevant_field(
    candidate,
    index,
    skill_count,
    matched_skills,
    career_score,
    behavior_score
):
    years = candidate["profile"]["years_of_experience"]

    title = "Engineer"
    if candidate["career_history"]:
        title = candidate["career_history"][0]["title"]

    sims = np.dot(
        reason_embeddings,
        career_emb[index]
    )
    field = reason_names[np.argmax(sims)]

    strengths = []

    if skill_count >= 6:
        strengths.append(
            f"{skill_count} JD-aligned skills"
        )

    if career_score > 0.60:
        strengths.append(
            f"strong {field} experience"
        )

    if behavior_score > 0.70:
        strengths.append(
            "strong recruiter engagement signals"
        )

    top_skills = matched_skills[:2]

    if len(top_skills) == 2:
        strengths.append(
            f"expertise in {top_skills[0]} and {top_skills[1]}"
        )
    elif len(top_skills) == 1:
        strengths.append(
            f"expertise in {top_skills[0]}"
        )

    concerns = []

    notice = candidate["redrob_signals"]["notice_period_days"]

    if notice >= 90:
        concerns.append(
            f"{notice}-day notice period"
        )

    if behavior_score < 0.4:
        concerns.append(
            "limited engagement signals"
        )

    strength_text = ", ".join(strengths)

    if concerns:
        return (
            f"{years:.1f} years of experience as a {title}. "
            f"Strong match through {strength_text}, "
            f"though {concerns[0]} may be a concern."
        )

    return (
        f"{years:.1f} years of experience as a {title}. "
        f"Strong match through {strength_text}."
    )