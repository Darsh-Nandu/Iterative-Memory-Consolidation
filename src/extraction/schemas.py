from pydantic import BaseModel, Field
from typing import List, Optional

class Fact(BaseModel):
    fact: str = Field(..., description="A singular, atomic proposition extracted from the text.")
    confidence: float = Field(..., description="Confidence score between 0.0 and 1.0.")
    source_chunk: int = Field(..., description="Index of the originating text chunk.")

class Entity(BaseModel):
    name: str = Field(..., description="Normalized canonical name of the entity.")
    type: str = Field(..., description="Classification category (e.g., PERSON, ORG, CONCEPT, EVENT).")

class Relationship(BaseModel):
    source: str = Field(..., description="Subject entity.")
    target: str = Field(..., description="Object entity.")
    relation: str = Field(..., description="The predicate mapping the relationship.")

class ExtractionPayload(BaseModel):
    facts: List[Fact]
    entities: List[Entity]
    relations: List[Relationship]

class CritiqueOutput(BaseModel):
    missing_facts: List[str] = Field(..., description="Crucial facts omitted from the memory.")
    weak_explanations: List[str] = Field(..., description="Vague or loosely represented ideas.")
    hallucinations: List[str] = Field(..., description="Claims found in the memory unsupported by ground truth.")

class QuestionAnswerPair(BaseModel):
    question: str
    answer: str
    type: str = Field(..., description="Type: factual, entity, reasoning, multi_hop, causal")

class QADatasetSchema(BaseModel):
    pairs: List[QuestionAnswerPair]