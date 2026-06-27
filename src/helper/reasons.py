from helper.profile_set import reason_names
import numpy as np

def _top_assessed_skill(candidate):
    """Pick the highest-scoring verified skill assessment, if any."""
    assessments = candidate["redrob_signals"].get("skill_assessment_scores", {})
    if not assessments:
        return None
    name, score = max(assessments.items(), key=lambda kv: kv[1])
    return (name, score) if score >= 70 else None


def _tenure_signal(candidate):
    """Flag stable long tenure vs. frequent job-hopping."""
    history = candidate["career_history"]
    if not history:
        return None
    current = history[0]
    months = current.get("duration_months") or 0
    if current.get("is_current") and months >= 24:
        return f"{months // 12}+ years in current role"
    if len(history) >= 3:
        avg_months = sum(h.get("duration_months") or 0 for h in history) / len(history)
        if avg_months < 12:
            return "short average tenure across past roles"
    return None


def _engagement_detail(candidate, behavior_score):
    """More specific than a generic behavior-score bucket."""
    sig = candidate["redrob_signals"]
    if behavior_score > 0.70:
        if sig.get("recruiter_response_rate", 0) >= 0.6:
            return f"high recruiter response rate ({sig['recruiter_response_rate']:.0%})"
        if sig.get("interview_completion_rate", 0) >= 0.6:
            return f"strong interview completion rate ({sig['interview_completion_rate']:.0%})"
        return "strong recruiter engagement signals"
    return None


def _practical_flags(candidate):
    """Logistics that affect close-ability, not match quality."""
    sig = candidate["redrob_signals"]
    flags = []
    if sig.get("applications_submitted_30d", 0) >= 20:
        flags.append("very high recent application volume")
    if sig.get("open_to_work_flag") is False:
        flags.append("not flagged as open to work")
    if sig.get("willing_to_relocate") is False:
        flags.append("not willing to relocate")
    return flags


def relevant_field(
    candidate,
    index,
    skill_count,
    matched_skills,
    career_score,
    behavior_score,
    career_emb,
    reason_embeddings
):
    years = candidate["profile"]["years_of_experience"]

    title = "Engineer"
    if candidate["career_history"]:
        title = candidate["career_history"][0]["title"]

    sims = np.dot(reason_embeddings, career_emb)
    field = reason_names[np.argmax(sims)]

    strengths = []

    if skill_count >= 6:
        strengths.append(f"{skill_count} JD-aligned skills")

    if career_score > 0.60:
        strengths.append(f"strong {field} experience")

    assessed = _top_assessed_skill(candidate)
    if assessed:
        strengths.append(f"verified {assessed[0]} assessment score of {assessed[1]:.0f}")

    engagement = _engagement_detail(candidate, behavior_score)
    if engagement:
        strengths.append(engagement)

    tenure = _tenure_signal(candidate)
    if tenure and "short" not in tenure:
        strengths.append(tenure)

    top_skills = matched_skills[:2]
    if len(top_skills) == 2:
        strengths.append(f"expertise in {top_skills[0]} and {top_skills[1]}")
    elif len(top_skills) == 1:
        strengths.append(f"expertise in {top_skills[0]}")

    concerns = []

    notice = candidate["redrob_signals"]["notice_period_days"]
    if notice >= 90:
        concerns.append(f"{notice}-day notice period")

    if behavior_score < 0.4:
        concerns.append("limited engagement signals")

    if tenure and "short" in tenure:
        concerns.append(tenure)

    concerns.extend(_practical_flags(candidate))

    strength_text = ", ".join(strengths) if strengths else f"general {field} background"

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