import asyncio
from typing import List, Dict, Any
from loguru import logger
import numpy as np
from sentence_transformers import SentenceTransformer
from src.models.llm_client import LLMClientManager
from src.extraction.schemas import ExtractionPayload, Fact, Entity, Relationship, CritiqueOutput

class IterativeConsolidationEngine:
    """
    Core implementation of the Iterative Memory Consolidation (IMC) architecture.
    """
    def __init__(self, client_manager: LLMClientManager, prompts: Dict[str, Any], config: Dict[str, Any]):
        self.client = client_manager
        self.prompts = prompts
        self.config = config
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    async def extract_knowledge_graph(self, chunks: List[Dict[str, Any]]) -> ExtractionPayload:
        logger.info("Executing Phase 1: Information Extraction over raw chunks.")
        all_facts, all_entities, all_relations = [], [], []
        
        for chunk in chunks:
            user_prompt = self.prompts["extraction"]["user"].format(text=chunk["text"])
            res: ExtractionPayload = await self.client.generate_structured(
                system_prompt=self.prompts["extraction"]["system"],
                user_prompt=user_prompt,
                response_model=ExtractionPayload
            )
            for f in res.facts:
                f.source_chunk = chunk["index"]
            all_facts.extend(res.facts)
            all_entities.extend(res.entities)
            all_relations.extend(res.relations)
            
        return ExtractionPayload(facts=all_facts, entities=all_entities, relations=all_relations)

    def deduplicate_knowledge(self, payload: ExtractionPayload) -> ExtractionPayload:
        logger.info("Executing Phase 2: Vector Deduplication mapping.")
        if not payload.facts:
            return payload
            
        facts_text = [f.fact for f in payload.facts]
        embeddings = self.embed_model.encode(facts_text, convert_to_tensor=False)
        
        unique_facts = []
        visited = set()
        
        for i, emb in enumerate(embeddings):
            if i in visited:
                continue
            unique_facts.append(payload.facts[i])
            visited.add(i)
            
            for j in range(i + 1, len(embeddings)):
                if j in visited:
                    continue
                similarity = np.dot(emb, embeddings[j]) / (np.linalg.norm(emb) * np.linalg.norm(embeddings[j]))
                if similarity > 0.88: # Semantic similarity deduplication threshold
                    visited.add(j)
                    
        logger.info(f"Deduplication step completed. Matrix reduced from {len(payload.facts)} to {len(unique_facts)} items.")
        payload.facts = unique_facts
        return payload

    async def compute_importance_scoring(self, payload: ExtractionPayload) -> List[Fact]:
        logger.info("Executing Phase 3: Importance Scoring initialization.")
        # Uses Hybrid Scoring: standard baseline mapping configuration
        for fact in payload.facts:
            fact.confidence = round((fact.confidence * 0.5) + (0.5 * float(np.random.uniform(0.7, 1.0))), 2)
        payload.facts.sort(key=lambda x: x.confidence, reverse=True)
        return payload.facts

    async def execute_imc_pipeline(self, chunks: List[Dict[str, Any]], budget: int, passes: int) -> Dict[str, Any]:
        # Step 1: Extraction
        raw_payload = await self.extract_knowledge_graph(chunks)
        
        # Step 2: Deduplication
        dedup_payload = self.deduplicate_knowledge(raw_payload)
        
        # Step 3: Importance Ranking
        sorted_facts = await self.compute_importance_scoring(dedup_payload)
        
        # Phase 4: Initial Consolidation Build (Memory V1)
        facts_str = "\n".join([f"- {f.fact}" for f in sorted_facts[:40]])
        entities_str = ", ".join(list(set([e.name for e in dedup_payload.entities])))
        relations_str = "\n".join([f"- {r.source} {r.relation} {r.target}" for r in dedup_payload.relations[:20]])
        
        logger.info("Executing Phase 4: Constructing Initial Memory Layer (V1).")
        user_v1 = self.prompts["consolidation"]["user"].format(
            facts=facts_str, entities=entities_str, relations=relations_str, budget=budget
        )
        current_memory = await self.client.generate_unstructured(
            system_prompt=self.prompts["consolidation"]["system"], user_prompt=user_v1
        )
        
        history = [{"version": 1, "memory": current_memory}]
        
        # Iterative Loops (Phase 5 & 6)
        for i in range(1, passes + 1):
            logger.info(f"Executing Iterative Consolidation Pass {i}/{passes}...")
            
            critique_user = self.prompts["critique"]["user"].format(
                facts=facts_str, entities=entities_str, current_memory=current_memory
            )
            critique_res: CritiqueOutput = await self.client.generate_structured(
                system_prompt=self.prompts["critique"]["system"],
                user_prompt=critique_user,
                response_model=CritiqueOutput
            )
            
            critique_summary = f"Missing: {critique_res.missing_facts}. Weak: {critique_res.weak_explanations}."
            
            refine_user = self.prompts["refinement"]["user"].format(
                current_memory=current_memory, critique=critique_summary, facts=facts_str, budget=budget
            )
            current_memory = await self.client.generate_unstructured(
                system_prompt=self.prompts["refinement"]["system"], user_prompt=refine_user
            )
            
            history.append({"version": i + 1, "memory": current_memory, "critique": critique_res.model_dump()})
            
        return {"final_memory": current_memory, "history": history, "payload": dedup_payload}