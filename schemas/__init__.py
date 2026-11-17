"""Pydantic models for evidence, diagnoses and review results, plus policy validation.

Every model here that can enter graph state must also be listed in
`app.state.CHECKPOINT_TYPES`; an unregistered type comes back from a checkpoint
as a plain dict rather than raising (T15).
"""
