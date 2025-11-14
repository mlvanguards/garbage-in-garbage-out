import json
from typing import Any, Dict, List, Optional

from src.db.manager import QDrantConnectionManager
from src.llm import LitellmClient
from src.prompts.answer_question import ANSWER_QUESTION_PROMPT_GEMINI
from src.prompts.query_decomposition import USER_QUESTION_DECOMPOSITION_PROMPT
from src.references import (
    TableReference,
    FigureReference,
    References,
    extract_tables_and_figures_references,
)
from src.references.extractor import (
    correlate_references_with_files as correlate_refs,
    deduplicate_references as deduplicate_refs,
)
from src.schemas import QueryDecompositionResponse
from src.config import settings




class QueryDecompositionService:
    """Service for decomposing user queries into sub-questions."""

    def __init__(self):
        self._llm_client = LitellmClient(model_name=settings.LLM_MODEL_NAME)

    def decompose_query(self, query: str) -> QueryDecompositionResponse:
        """
        Decompose a user query into sub-questions.
        
        Args:
            query: The user's original question
            
        Returns:
            QueryDecompositionResponse containing decomposed questions
        """
        resp = self._llm_client.chat(
            messages=[
                {
                    "role": "system",
                    "content": "You are a technical assistant that decomposes a user's question into a list of sub-questions.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": USER_QUESTION_DECOMPOSITION_PROMPT.format(
                                user_question=query
                            ),
                        },
                    ],
                },
            ],
            response_format=QueryDecompositionResponse,
        )
        
        # Parse the response
        content = resp.choices[0].message.content
        if isinstance(content, str):
            return QueryDecompositionResponse.model_validate_json(content)
        return content
    


class RetrievalService:
    """Service for retrieving relevant points and answering questions."""

    def __init__(
        self,
        retriever_strategy: Optional[Any] = None,
        collection_name: Optional[str] = None,
    ):
        """
        Initialize the retrieval service.
        
        Args:
            retriever_strategy: Optional retriever strategy (BaseQdrantRetriever instance).
                               If None, will use QDrantConnectionManager directly.
            collection_name: Name of the Qdrant collection to query
        """
        self.db_manager = QDrantConnectionManager()
        self._query_decomposition_service = QueryDecompositionService()
        self._llm_client = LitellmClient(model_name=settings.LLM_MODEL_NAME)
        self._retriever_strategy = retriever_strategy
        self._collection_name = collection_name or "hybrid_collection"

    def answer_question(self, user_question: str) -> Dict[str, Any]:
        """
        Answer a user question by decomposing it, retrieving relevant points, and generating an answer.
        
        Args:
            user_question: The user's question to answer
            
        Returns:
            Dictionary with 'answer' and 'references' keys
        """
        # Retrieve relevant points for the question
        relevant_points = self._retrieve_points(user_question)
        
        # Extract references from relevant points
        references = extract_tables_and_figures_references(relevant_points)

        # Get the answer
        resp = self._llm_client.chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a technical assistant that answers a user's "
                        "question based on the relevant points from a service manual."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": ANSWER_QUESTION_PROMPT_GEMINI.format(
                                user_question=user_question,
                                relevant_points=relevant_points,
                            ),
                        },
                    ],
                },
            ],
            response_format=None,
            temperature=0.0,
            max_tokens=10000,
        )
        answer = resp.choices[0].message.content

        return {"answer": answer, "references": references}

    def _retrieve_points(self, query: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve relevant points by decomposing the query and retrieving for each sub-question.
        
        Args:
            query: The user's query
            
        Returns:
            Dictionary mapping sub-questions to their retrieved results
        """
        # Decompose the query into sub-questions
        decomposition = self._query_decomposition_service.decompose_query(query)
        
        results: Dict[str, List[Dict[str, Any]]] = {}

        # Retrieve points for each sub-question
        for sub_question_mapping in decomposition.decomposed_questions:
            sub_question = sub_question_mapping.sub_question
            # Use the retriever strategy if available, otherwise use direct Qdrant client
            if self._retriever_strategy:
                results[sub_question] = self._retriever_strategy.retrieve(
                    query=sub_question,
                    collection_name=self._collection_name
                )
            else:
                # Fallback: use direct Qdrant client (basic implementation)
                # This would need to be implemented based on your specific needs
                results[sub_question] = self._retrieve_with_direct_client(sub_question)
        
        return results
