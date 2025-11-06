import json

from src.processors.base import BaseProcessor
from src.prompts.question_mapping import MAP_QUESTION_SECTION_PROMPT
from src.llm import LitellmClient
from src.schemas import QuestionMappingResponse


class QuestionMappingProcessor(BaseProcessor):

    def __init__(self):
        self._llm_client = LitellmClient(model_name="openai/gpt-4o")

    def process(self, question: str) -> str:
        resp = self._llm_client.chat(
            messages=[
                {
                    "role": "system",
                    "content": "You are a technical assistant that maps a user's question to the most relevant section and chapter in a service manual.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": MAP_QUESTION_SECTION_PROMPT.format(
                                user_question=user_question
                            ),
                        },
                    ],
                },
            ],
            response_format=QuestionMappingResponse,
        )
        return json.loads(resp.choices[0].message.content)


if __name__ == "__main__":
    processor = QuestionMappingProcessor()
    user_question = "What sequence of checks should be performed if the engine starts but none of the hydraulic functions work?"
    print(processor.process(user_question))