import os
import asyncio
from typing import Type, Any, Dict
from loguru import logger
from pydantic import BaseModel
from google import genai
from google.genai import types

class LLMClientManager:
    """
    Unified execution abstraction layer handling Google Gemini API native
    Structured Outputs with built-in rate-limiting mitigations for Free Tiers.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config["model"]
        self.model_name = self.config.get("name", "gemini-2.5-flash")
        
        # Initialize the modern official Google GenAI Client
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is missing.")
        self.client = genai.Client(api_key=api_key)

    async def _rate_limit_backoff(self):
        """Free tier structural cooling-off safety buffer (15 RPM mitigation)"""
        await asyncio.sleep(4.0)

    async def generate_structured(self, system_prompt: str, user_prompt: str, response_model: Type[BaseModel]) -> BaseModel:
        await self.rate_limit_backoff()
        try:
            # We wrap execution in an async executor thread since the current 
            # SDK's structural configuration uses a synchronous engine call architecture.
            loop = asyncio.get_running_loop()
            
            def call_gemini():
                return self.client.models.generate_content(
                    model=self.model_name,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=self.config.get("temperature", 0.0),
                        response_mime_type="application/json",
                        response_schema=response_model,
                    ),
                )

            response = await loop.run_in_executor(None, call_gemini)
            # Use Pydantic's native validation to parse Gemini's text response string
            return response_model.model_validate_json(response.text)
            
        except Exception as e:
            logger.error(f"Error communicating with Gemini Structured Pipeline: {e}")
            raise e

    async def generate_unstructured(self, system_prompt: str, user_prompt: str) -> str:
        await self._rate_limit_backoff()
        try:
            loop = asyncio.get_running_loop()
            
            def call_gemini():
                return self.client.models.generate_content(
                    model=self.model_name,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=self.config.get("temperature", 0.0),
                    ),
                )

            response = await loop.run_in_executor(None, call_gemini)
            return response.text
        except Exception as e:
            logger.error(f"Error during unstructured generation payload on Gemini: {e}")
            raise e