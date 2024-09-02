from typing import List
import dspy
from difflib import SequenceMatcher
from langchain.text_splitter import TokenTextSplitter


def split_into_chunks(text: str, chunk_size: int) -> List[str]:
    text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    return text_splitter.split_text(text)


def calculate_utilization_score(attributed_text: str, model_answer: str) -> float:
    matcher = SequenceMatcher(None, attributed_text, model_answer)
    intersection_length = sum(length for _, _, length in matcher.get_matching_blocks())
    return intersection_length / len(model_answer) if len(model_answer) > 0 else 0.0


class ChunkAnalysis(dspy.Signature):
    """Analyzes each text chunk to determine its influence on the model's generated response (attribution)."""

    user_question: str = dspy.InputField(
        desc="The original question or query input by the user.")

    context: List[str] = dspy.InputField(
        desc="A segment of text retrieved by the search engine that is relevant to the user's question.")

    answer: str = dspy.InputField(
        desc="The response produced by the model after processing the user's question in conjunction with the provided context.")

    analysis_result: bool = dspy.OutputField(
        desc="Indicates whether the specific text chunk influenced the model's response. True if it did, otherwise False.")
