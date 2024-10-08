from typing import Dict, Any, List, Optional
from enum import Enum
from distutils.util import strtobool
from chunk_based_metrics import ChunkAnalysis, split_into_chunks
from text_based_calculations import calculate_sbert_similarity
from pydantic import BaseModel
from loguru import logger
import dspy


class EvaluationMetric(str, Enum):
    CONTEXT_ADHERENCE = "Context Adherence"
    CORRECTNESS = "Correctness"
    COMPLETENESS = "Completeness"
    INSTRUCTION_ADHERENCE = "Instruction Adherence"
    CONTEXT_RELEVANCE = "Context Relevance"
    CONCISENESS_AND_RELEVANCE = "conciseness and relevance"


class MetricDescription(str, Enum):
    CONTEXT_ADHERENCE_DESC = (
        "Task: Evaluate if the answer strictly uses only the information provided in the context (precision).\n"
        "Steps:\n"
        "1. Carefully read the context and the answer.\n"
        "2. Identify any information in the answer that is not present in the context.\n"
        "3. If the answer contains only information from the context, evaluate as 'passed'.\n"
        "4. If the answer includes any information not in the context, evaluate as 'failed'.\n"
        "Note: This metric measures closed-domain hallucinations. A score of 1 or close to 1 indicates high adherence to the context.\n"
    )

    CORRECTNESS_DESC = (
        "Task: Assess the factual accuracy of the answer based on the domain expert answer provided.\n"
        "Steps:\n"
        "1. Compare each statement in the answer to the information in the expert answer.\n"
        "2. Check for any factual errors or contradictions between the model answer and the expert answer.\n"
        "3. If all statements in the answer are factually correct and align with the expert answer, evaluate as 'passed'.\n"
        "4. If any statement is incorrect or contradicts the expert answer, evaluate as 'failed'.\n"
    )

    COMPLETENESS_DESC = (
        "Task: Determine if the answer fully addresses all aspects of the user's question using the context provided, focusing on recall of relevant information.\n"
        "Steps:\n"
        "1. Identify all parts of the user's question.\n"
        "2. Check if the answer addresses each part of the question.\n"
        "3. Verify that all relevant information from the context is included in the answer, even if not explicitly asked for in the question.\n"
        "4. Assess whether the response fully reflects the relevant information available in the context.\n"
        "5. If the answer covers all parts of the question, includes all relevant context, and doesn't omit important information, evaluate as 'passed'.\n"
        "6. If any part of the question is unanswered, relevant context is missing, or important information is omitted, evaluate as 'failed'.\n"
        "Note: This metric complements Context Adherence. While Context Adherence focuses on precision, Completeness focuses on recall.\n"
    )
    INSTRUCTION_ADHERENCE_DESC = (
        "Task: Evaluate if the answer follows the specific instructions or requirements in the user's question or system prompt.\n"
        "Steps:\n"
        "1. Identify any specific instructions or requirements in the user's question or system prompt.\n"
        "2. Check if the answer addresses these specific instructions or requirements.\n"
        "3. If the answer follows all instructions and meets all requirements, evaluate as 'passed'.\n"
        "4. If the answer fails to follow any instruction or meet any requirement, evaluate as 'failed'.\n"
        "Note: This metric helps uncover hallucinations where the model is ignoring instructions. A high score (close to 1) indicates the model likely followed its instructions.\n"
    )
    CONCISENESS_AND_RELEVANCE_DESC = (
        "Task: Evaluate if the answer includes only information relevant to the question, without adding unnecessary details from the context.\n"
        "Steps:\n"
        "1. Carefully read the user's question and the provided answer.\n"
        "2. Identify the main points and requirements of the question.\n"
        "3. Check if each piece of information in the answer directly addresses the question or is necessary for a complete understanding.\n"
        "4. Identify any information from the context that is included in the answer but not relevant to the specific question asked.\n"
        "5. If the answer contains only relevant information and does not include unnecessary details from the context, evaluate as 'passed'.\n"
        "6. If the answer includes irrelevant information or unnecessary details from the context, even if factually correct, evaluate as 'failed'.\n"
        "Note: This metric helps ensure the model's response is focused, on-topic, and concise. A high score (close to 1) indicates the answer is relevant to the question asked without including extraneous information.\n"
    )


class MetricDetails(BaseModel):
    metric: EvaluationMetric
    description: MetricDescription


class RAGEvaluatorWithHumanAnswer(dspy.Signature):
    """Signature for the RAG evaluator."""
    user_question: str = dspy.InputField(desc="The question entered by the user.")
    context: List[str] = dspy.InputField(
        desc="The chunks of text retrieved by the search engine related to the user's question.")
    answer: str = dspy.InputField(
        desc="The response generated by the model based on the user's question and the provided context.")
    metric: MetricDetails = dspy.InputField(desc="The specific aspect of the response being evaluated.")
    human_answer: str = dspy.InputField(desc="A human answer provided by a domain expert in the field.")
    evaluation_result: str = dspy.OutputField(
        desc="The outcome of the evaluation process, did the response meet the criteria for the selected metric? output a final result ('failed' or 'passed').")


class RAGEvaluator(dspy.Signature):
    """Signature for the RAG evaluator."""
    user_question: str = dspy.InputField(desc="The question entered by the user.")
    context: List[str] = dspy.InputField(
        desc="The chunks of text retrieved by the search engine related to the user's question.")
    answer: str = dspy.InputField(
        desc="The response generated by the model based on the user's question and the provided context.")
    metric: MetricDetails = dspy.InputField(desc="The specific aspect of the response being evaluated.")
    evaluation_result: str = dspy.OutputField(
        desc="The outcome of the evaluation process, did the response meet the criteria for the selected metric? output a final result ('failed' or 'passed').")


class RAGEvaluatorWithSystemPrompt(dspy.Signature):
    """Signature for the RAG evaluator."""
    system_prompt: str = dspy.InputField(desc="The system prompt assigned to the model.")
    user_question: str = dspy.InputField(desc="The question entered by the user.")
    context: List[str] = dspy.InputField(
        desc="The chunks of text retrieved by the search engine related to the user's question.")
    answer: str = dspy.InputField(
        desc="The response generated by the model based on the user's question and the provided context.")
    metric: MetricDetails = dspy.InputField(desc="The specific aspect of the response being evaluated.")
    evaluation_result: str = dspy.OutputField(
        desc="The outcome of the evaluation process, did the response meet the criteria for the selected metric? output a final result ('failed' or 'passed').")


class EvalParser(dspy.Signature):
    """Parse the evaluation result to return a boolean value. 'passed' maps to True, and 'failed' maps to False."""
    models_output = dspy.InputField(desc="The evaluation result containing the final result and explanation.")
    formatted_output = dspy.OutputField(
        desc="The final result of the model's output. return False if 'failed', True if 'passed'.")


class RAGEval(dspy.Module):
    """
    Module for RAG (Retrieval-Augmented Generation) evaluation.

    This class provides methods to evaluate RAG outputs based on various metrics,
    including correctness, instruction adherence, and chunk-level analysis.
    """

    def __init__(self):
        """
        Initialize the RAGEval module with different evaluation strategies.
        """
        super().__init__()
        logger.info("Initializing RAGEval module")
        self.generate_eval = dspy.TypedChainOfThought(RAGEvaluator)
        self.generate_eval_with_human_answer = dspy.TypedChainOfThought(RAGEvaluatorWithHumanAnswer)
        self.generate_eval_with_system_prompt = dspy.TypedChainOfThought(RAGEvaluatorWithSystemPrompt)
        self.generate_chunk_level_eval = dspy.TypedChainOfThought(ChunkAnalysis)
        self.format_eval = dspy.ChainOfThought(EvalParser)
        logger.debug("RAGEval module initialized successfully")

    def forward(self, user_question: str, context: List[str], answer: str, metric: MetricDetails,
                human_answer: Optional[str] = None, system_prompt: Optional[str] = None) -> dspy.Prediction:
        """
        Perform the forward pass of the RAG evaluation.

        Args:
            user_question (str): The original question asked by the user.
            context (List[str]): The context provided for answering the question.
            answer (str): The generated answer to evaluate.
            metric (MetricDetails): Details of the evaluation metric to be used.
            human_answer (Optional[str]): The human-generated answer, if available.
            system_prompt (Optional[str]): The system prompt used for generation, if available.

        Returns:
            Prediction: A dspy.Prediction object containing the final judgment as a boolean.

        Raises:
            ValueError: If an unsupported evaluation metric is provided.
        """
        logger.info(f"Starting evaluation for metric: {metric.metric}")
        logger.debug(f"User question: {user_question}")
        logger.debug(f"Answer to evaluate: {answer}")
        logger.debug(f"Metric details: {metric}")

        # Choose the appropriate evaluation method based on the metric and available inputs
        if metric.metric == EvaluationMetric.CORRECTNESS and human_answer:
            logger.info("Using evaluation with human answer")
            eval_result = self.generate_eval_with_human_answer(
                user_question=user_question,
                context=context,
                answer=answer,
                metric=metric,
                human_answer=human_answer
            ).evaluation_result
        elif metric.metric == EvaluationMetric.INSTRUCTION_ADHERENCE and system_prompt:
            logger.info("Using evaluation with system prompt")
            eval_result = self.generate_eval_with_system_prompt(
                system_prompt=system_prompt,
                user_question=user_question,
                context=context,
                answer=answer,
                metric=metric
            ).evaluation_result
        else:
            logger.info(f"Using standard evaluation for {metric.metric}")
            eval_result = self.generate_eval(
                user_question=user_question,
                context=context,
                answer=answer,
                metric=metric
            ).evaluation_result

        logger.debug(f"Raw evaluation result: {eval_result}")

        # Format the evaluation result
        logger.info("Formatting evaluation result")
        model = dspy.OpenAI(temperature=0.0)
        dspy.settings.configure(lm=model)
        formatted_eval_result = self.format_eval(models_output=str(eval_result)).formatted_output
        final_judgment = bool(strtobool(formatted_eval_result))
        logger.info(f"Final judgment: {final_judgment}")

        return dspy.Prediction(final_judgment=final_judgment)


def evaluate_responses(user_question: str, context: str, answer: str, human_answer: str, system_prompt: str,
                       metrics: List[MetricDetails],
                       num_evaluations: int = 5) -> Dict[str, Any]:
    """Evaluate responses for all metrics with varying temperatures, considering majority-vote attribution."""

    logger.info(f"Starting evaluation for all metrics")
    results = {metric.metric.value: [] for metric in metrics}
    chunk_attribution_votes = {}

    for i in range(num_evaluations):
        temperature = i / (num_evaluations - 1)  # Adjust temperature for each iteration
        logger.debug(f"Evaluation {i + 1}/{num_evaluations} with temperature: {temperature}")

        # Reconfigure the OpenAI model with the new temperature
        model = dspy.OpenAI(temperature=temperature)
        dspy.settings.configure(lm=model)

        # Create a new RAGEval instance after reconfiguring the model
        evaluator = RAGEval()

        # Generate chunk-level evaluation once per temperature setting
        chunks = split_into_chunks(context, 128)
        logger.debug(f"Split context into {len(chunks)} chunks")

        for chunk in chunks:
            logger.debug(f"Evaluating chunk: '{chunk}...'")
            chunk_result = evaluator.generate_chunk_level_eval(
                user_question=user_question,
                context=[chunk],
                answer=answer
            ).analysis_result

            # Log the chunk attribution result with improved clarity
            impact_label = "high impact" if chunk_result else "low impact"
            logger.debug(
                f"Chunk labeled as {impact_label}.")

            # Track attribution votes for each chunk
            if chunk not in chunk_attribution_votes:
                chunk_attribution_votes[chunk] = 0
            if chunk_result:
                chunk_attribution_votes[chunk] += 1

        # Evaluate each metric using the same chunk-level evaluation
        for metric in metrics:
            result = evaluator(user_question, chunks, answer, metric, human_answer, system_prompt).final_judgment
            results[metric.metric.value].append(result)
            logger.debug(f"Evaluation {i + 1} result for {metric.metric.value}: {result}")

    # Determine which chunks had high attribution by majority vote
    majority_attributed_chunks = [
        chunk for chunk, votes in chunk_attribution_votes.items()
        if votes > num_evaluations / 2  # Majority vote threshold
    ]
    logger.debug(f"{len(majority_attributed_chunks)} chunks determined to have high attribution by majority vote")

    # Calculate attribution scores for chunks with high attribution
    attributed_chunk_utilization_dict = {}
    for chunk in majority_attributed_chunks:
        chunk_utilization_score = calculate_sbert_similarity(chunk, answer)
        attributed_chunk_utilization_dict[chunk] = chunk_utilization_score
        logger.debug(f"Chunk: '{chunk}...' - Utilization Score: {chunk_utilization_score:.4f}")

    # Combine results and separate attribution scores
    combined_results = {
        "metric_results": {},
        "chunk_attribution_scores": attributed_chunk_utilization_dict
    }

    for metric, metric_results in results.items():
        if metric_results:
            average_score = sum(r for r in metric_results) / len(metric_results)
            combined_results["metric_results"][metric] = {
                "average_score": average_score,
                "individual_results": [r for r in metric_results]
            }
            logger.info(f"Evaluation complete for {metric}. Average score: {average_score:.2f}")
        else:
            combined_results["metric_results"][metric] = {"error": "Failed to evaluate this metric"}

    return combined_results


def evaluate_all_metrics(user_question: str, context: str, answer: str, human_answer: str, system_prompt: str,
                         num_evaluations: int = 5) -> \
        Dict[str, Any]:
    """Evaluate all metrics for a given question, context, and answer with varying temperatures."""
    logger.info("Starting evaluation of all metrics")
    metrics = [
        MetricDetails(metric=EvaluationMetric.CONTEXT_ADHERENCE, description=MetricDescription.CONTEXT_ADHERENCE_DESC),
        MetricDetails(metric=EvaluationMetric.CORRECTNESS, description=MetricDescription.CORRECTNESS_DESC),
        MetricDetails(metric=EvaluationMetric.COMPLETENESS, description=MetricDescription.COMPLETENESS_DESC),
        MetricDetails(metric=EvaluationMetric.INSTRUCTION_ADHERENCE,
                      description=MetricDescription.INSTRUCTION_ADHERENCE_DESC),
        MetricDetails(metric=EvaluationMetric.CONCISENESS_AND_RELEVANCE,
                      description=MetricDescription.CONCISENESS_AND_RELEVANCE_DESC)
    ]

    try:
        # Call the evaluate_responses function to get scores and attribution
        scores = evaluate_responses(user_question, context, answer, human_answer, system_prompt, metrics,
                                    num_evaluations)

    except Exception as e:
        logger.error(f"Error evaluating metrics: {str(e)}")
        logger.exception("Full traceback:")
        scores = {"error": str(e)}

    return scores
