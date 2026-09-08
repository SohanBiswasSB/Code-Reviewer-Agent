from registry import surface

@surface("code_review")
class CodeReviewSurface:
    name = "code_review"
    source_name, sink_name = "github", "pr_comment"      # resolved by config
    diagnosis_prompt = "prompts/code_review_diagnosis.md"
    judge_prompt = "prompts/code_review_judge.md"

    def build_query(self, evidence: dict) -> str:
        return f"{evidence.get('title','')}\n{evidence.get('diff','')}"

    def classify(self, evidence: dict) -> str:
        risky = {"migrations", "auth", "permission", "credly", "celery", "routing"}
        touched = set(evidence.get("touched_areas", []))
        if evidence.get("has_failing_ci"):
            return "deterministic"
        if touched & risky:
            return "ai_investigation"
        return "human_review"
