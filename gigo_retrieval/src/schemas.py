from pydantic import BaseModel
from typing import List

class SubQuestionMapping(BaseModel):
    sub_question: str
    section_number: int
    section_title: str
    matched_chapters: List[str]

class QueryDecompositionResponse(BaseModel):
    original_question: str
    decomposed_questions: List[SubQuestionMapping]