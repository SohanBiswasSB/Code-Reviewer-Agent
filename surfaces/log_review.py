from registry import surface

@surface("log_review")
class LogReviewSurface:
    name = "log_review"
    source_name, sink_name = "s3logs", "report"
    diagnosis_prompt = "prompts/log_review_analysis.md"
    judge_prompt = "prompts/log_review_judge.md"

    def build_query(self, evidence: dict) -> str:
        sigs = ", ".join(c.get("signal", "") for c in evidence.get("candidates", []))
        return f"anomalies: {sigs}"

    def classify(self, evidence: dict) -> str:
        cands = evidence.get("candidates", [])
        if not cands:
            return "human_review"          # nothing above threshold -> healthy report
        return "ai_investigation"
