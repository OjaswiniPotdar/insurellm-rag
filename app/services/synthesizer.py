from typing import List
import pandas as pd
from pydantic import BaseModel, Field
from app.services.llm_factory import LLMFactory


class SynthesizedResponse(BaseModel):
    thought_process: List[str] = Field(
        description="Reasoning steps used to synthesize the answer"
    )
    answer: str = Field(description="Final answer to the user question")
    enough_context: bool = Field(
        description="Whether retrieved context is sufficient"
    )


class Synthesizer:

    SYSTEM_PROMPT = """
# Role
You are an AI assistant for **Insurellm**, an insurance company.

Your job is to answer questions using information retrieved from the company's internal knowledge base.

# Knowledge Base Contains
The retrieved context may include information about:
- Insurellm employees and organizational structure
- Company policies and internal information
- Insurance contracts and legal documents
- Insurance products and services offered by Insurellm

# Instructions
1. Answer the user's question clearly and professionally.
2. Use ONLY the information provided in the retrieved context.
3. The context is retrieved using vector similarity, so some parts may be irrelevant.
4. Focus only on the information that directly helps answer the question.
5. If the retrieved context does not contain enough information, clearly state that.
6. Do NOT fabricate or assume information that is not present in the context.
7. When relevant, mention the source document or section from the retrieved information.
8. Maintain a professional tone suitable for insurance and business communication.
9. If the context contains structured information (such as feature lists, numbered items, or bullet points), present them clearly instead of summarizing them.

# Goal
Provide accurate, trustworthy answers based strictly on Insurellm's internal knowledge base.
"""

    @staticmethod
    def generate_response(question: str, context: pd.DataFrame) -> SynthesizedResponse:

        context_str = Synthesizer.dataframe_to_json(
            context,
            columns_to_keep=["contents"]
        )

        messages = [
            {
                "role": "system",
                "content": Synthesizer.SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"User question:\n{question}"
            },
            {
                "role": "assistant",
                "content": f"Retrieved context:\n{context_str}"
            }
        ]

        llm = LLMFactory("openai")

        return llm.create_completion(
            response_model=SynthesizedResponse,
            messages=messages,
        )

    @staticmethod
    def dataframe_to_json(
        context: pd.DataFrame,
        columns_to_keep: List[str],
    ) -> str:

        return context[columns_to_keep].to_json(
            orient="records",
            indent=2
        )