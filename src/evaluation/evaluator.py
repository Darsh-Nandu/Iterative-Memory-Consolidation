import asyncio
from typing import List, Dict, Any
from loguru import logger
from src.models.llm_client import LLMClientManager
from src.extraction.schemas import QADatasetSchema, QuestionAnswerPair
from src.evaluation.metrics import QuantitativeMetricsSuite

class EvaluationEngine:
    """
    Orchestrates full synthetic verification frameworks, baseline executions, and metrics generation.
    """
    def __init__(self, client_manager: LLMClientManager, prompts: Dict[str, Any]):
        self.client = client_manager
        self.prompts = prompts
        self.metrics_suite = QuantitativeMetricsSuite()

    async def generate_synthetic_qa(self, raw_text: str, num_questions: int) -> List[QuestionAnswerPair]:
        logger.info(f"Generating {num_questions} synthetic execution verification evaluation items.")
        user_prompt = self.prompts["qa_generation"]["user"].format(count=num_questions, text=raw_text[:8000])
        res: QADatasetSchema = await self.client.generate_structured(
            system_prompt=self.prompts["qa_generation"]["system"],
            user_prompt=user_prompt,
            response_model=QADatasetSchema
        )
        return res.pairs

    async def evaluate_memory_system(self, context: str, qa_pairs: List[QuestionAnswerPair]) -> List[Dict[str, Any]]:
        results = []
        for pair in qa_pairs:
            system_prompt = "You are a precise QA bot. Answer the question using ONLY the provided context snippet."
            user_prompt = f"Context Material:\n{context}\n\nQuestion: {pair.question}"
            
            predicted_answer = await self.client.generate_unstructured(system_prompt, user_prompt)
            
            f1 = self.metrics_suite.compute_token_f1(predicted_answer, pair.answer)
            sim = self.metrics_suite.compute_semantic_similarity(predicted_answer, pair.answer)
            
            results.append({
                "question": pair.question,
                "ground_truth": pair.answer,
                "prediction": predicted_answer,
                "f1_score": f1,
                "semantic_similarity": sim
            })
        return results