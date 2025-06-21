from typing import List, Literal
from pydantic import BaseModel, Field


class TextComparisonResult(BaseModel):
    """Data model for the combined textual analysis."""
    SEMANTIC_DIFF_STATUS: Literal["IDENTICAL", "MINOR_DIFFERENCE", "VAST_DIFFERENCE"] = Field(
        ..., description="Classification of the difference in meaning and intent."
    )
    SEMANTIC_DIFF_SUMMARY: List[str] = Field(
        ..., description="A summary of differences in meaning, key facts, or calls-to-action."
    )

class VisualComparisonResult(BaseModel):
    """Data model for the visual analysis."""
    VISUAL_DIFF_STATUS: Literal["IDENTICAL", "MINOR_DIFFERENCE", "VAST_DIFFERENCE"] = Field(
        ..., description="Classification of the visual difference in layout, style, or imagery."
    )
    VISUAL_DIFF_SUMMARY: List[str] = Field(
        ..., description="A summary of detected visual differences."
    )

class TextDiffResult(BaseModel):
    """Data model for the literal text diff analysis."""
    TEXT_DIFF_STATUS: Literal["IDENTICAL", "MINOR_DIFFERENCE", "VAST_DIFFERENCE"] = Field(
        ..., description="Classification of the literal, character-by-character text difference."
    )
    TEXT_DIFF_SUMMARY: List[str] = Field(
        ..., description="A summary of exact wording, punctuation, or whitespace changes, like a git diff."
    )
