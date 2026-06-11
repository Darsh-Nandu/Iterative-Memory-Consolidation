import os
import asyncio
from typing import Type, Any, Dict
from loguru import logger
from pydantic import BaseModel
import ollama  # Import the local Ollama client wrapper

class LLMClientManager:
    """
    Unified client layer routing generations to either Google Cloud GenAI 
    or a local Ollama instance with full native Pydantic schema validation.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config["model"]
        self.provider = self.config.get("provider", "google").lower()
        self.model_name = self.config.get("name", "gemma4:e2b")
        
        if self.provider == "google":
            from google import genai
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable is missing.")
            self.google_client = genai.Client(api_key=api_key)
        else:
            # Initialize Ollama asynchronous internal client pipeline
            self.ollama_client = ollama.AsyncClient()

    async def generate_structured(self, system_prompt: str, user_prompt: str, response_model: Type[BaseModel]) -> BaseModel:
        if self.provider == "google":
            # [Optional Cloud Track] Keep old Google logic if you switch back
            await asyncio.sleep(4.0) # Free tier mitigation fallback
            loop = asyncio.get_running_loop()
            from google.genai import types
            def call_gemini():
                return self.google_client.models.generate_content(
                    model=self.model_name, contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=self.config.get("temperature", 0.0),
                        response_mime_type="application/json", response_schema=response_model,
                    ),
                )
            response = await loop.run_in_executor(None, call_gemini)
            return response_model.model_validate_json(response.text)
            
        else:
            # Local Ollama GBNF Grammar Constrained Structured Generation Loop
            try:
                response = await self.ollama_client.chat(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": f"{system_prompt}\nYou MUST respond strictly matching the requested JSON format."},
                        {"role": "user", "content": user_prompt}
                    ],
                    options={"temperature": self.config.get("temperature", 0.0)},
                    format=response_model.model_json_schema() # Dynamically passes Pydantic down as JSON Schema
                )
                raw_content = response.message.content
                return response_model.model_validate_json(raw_content)
            except Exception as e:
                logger.error(f"Ollama local structured generation failed: {e}")
                raise e

    async def generate_unstructured(self, system_prompt: str, user_prompt: str) -> str:
        if self.provider == "google":
            await asyncio.sleep(4.0)
            loop = asyncio.get_running_loop()
            from google.genai import types
            def call_gemini():
                return self.google_client.models.generate_content(
                    model=self.model_name, contents=user_prompt,
                    config=types.GenerateContentConfig(system_instruction=system_prompt, temperature=self.config.get("temperature", 0.0)),
                )
            response = await loop.run_in_executor(None, call_gemini)
            return response.text
        else:
            try:
                response = await self.ollama_client.chat(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    options={"temperature": self.config.get("temperature", 0.0)}
                )
                return response.message.content
            except Exception as e:
                logger.error(f"Ollama unstructured generation failure: {e}")
                raise e