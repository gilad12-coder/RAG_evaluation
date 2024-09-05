import time
from typing import Dict, Any
from statistics import mean, stdev

import nltk
from loguru import logger
import pandas as pd
import phoenix as px
from openinference.instrumentation.dspy import DSPyInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from text_based_calculations import calculate_textual_and_semantic_correlation_metrics

from config.logger_config import configure_logger
from utils.openai_utils import initialize_openai_model
from RAG_evaluator import evaluate_all_metrics

# Configure logger
configure_logger()

# Launch the Phoenix application
px.launch_app()

# Ensure necessary NLTK data is downloaded
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)


# Configure OpenTelemetry tracing
def configure_tracing():
    endpoint = "http://127.0.0.1:6006/v1/traces"
    resource = Resource(attributes={})
    tracer_provider = trace_sdk.TracerProvider(resource=resource)
    span_otlp_exporter = OTLPSpanExporter(endpoint=endpoint)
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_otlp_exporter))
    trace_api.set_tracer_provider(tracer_provider)

    # Instrument DSPy
    DSPyInstrumentor().instrument()


configure_tracing()

# Initialize OpenAI model
model = initialize_openai_model()


def expand_dict_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expands dictionary columns in a DataFrame into separate columns.

    This function is designed for a DataFrame with columns that contain dictionaries,
    specifically for 'scores', 'chunk_attribution_scores', and 'metadata'. The dictionaries
    in these columns are expanded into individual columns, and the original dictionary
    columns are dropped from the DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing columns with dictionary values.

    Returns:
    --------
    pd.DataFrame
        A new DataFrame with the dictionary columns expanded into individual columns.
    """
    # Expand the 'scores' column into separate columns
    scores_df = df['scores'].apply(pd.Series)

    # Expand the 'metadata' column into separate columns
    metadata_df = df['metadata'].apply(pd.Series)

    # Drop the original dictionary columns from the DataFrame
    df_cleaned = df.drop(['scores', 'metadata'], axis=1)

    # Concatenate the expanded DataFrames back to the original DataFrame
    df_final = pd.concat([df_cleaned, scores_df, metadata_df], axis=1)

    return df_final


def evaluate_response(row: pd.Series, num_evaluations: int = 5) -> Dict[str, Any]:
    start_time = time.time()
    logger.info(f"Starting evaluation for question: '{row['question'][:50]}...'")

    combined_scores = {}
    chunk_attribution_scores = {}

    try:
        # Evaluate using text-based metrics
        logger.debug("Calculating text-based metrics")
        text_metrics_start = time.time()
        text_metrics = calculate_textual_and_semantic_correlation_metrics(row['human_answer'], row['model_answer'])
        text_metrics_time = time.time() - text_metrics_start
        logger.debug(f"Text-based metrics calculation completed in {text_metrics_time:.2f} seconds")

        # Evaluate using RAG evaluator
        logger.debug("Calculating RAG evaluation scores")
        rag_start = time.time()
        rag_scores = evaluate_all_metrics(row['question'], row['context'], row['model_answer'], row["human_answer"], row["system_prompt"], num_evaluations)
        rag_time = time.time() - rag_start
        logger.debug(f"RAG evaluation completed in {rag_time:.2f} seconds")

        # Combine all scores
        combined_scores = {**text_metrics, **rag_scores.get("metric_results", {})}
        chunk_attribution_scores = rag_scores.get("chunk_attribution_scores", {})

        total_time = time.time() - start_time
        logger.success(f"Evaluation completed for the response in {total_time:.2f} seconds")

    except Exception as e:
        logger.error(f"Error in evaluate_response: {str(e)}")
        logger.exception("Full traceback:")
        combined_scores = {"error": str(e)}

    finally:
        # Always return a dictionary, even if it's empty or contains an error message
        return {
            "scores": combined_scores,
            "chunk_attribution_scores": chunk_attribution_scores,
            "metadata": {
                "question_length": len(row['question']),
                "context_length": len(row['context']),
                "model_answer_length": len(row['model_answer']),
                "human_answer_length": len(row['human_answer']),
                "num_evaluations": num_evaluations,
                "total_time": time.time() - start_time
            }
        }


def evaluate_dataset(dataset: pd.DataFrame, num_evaluations: int = 5) -> pd.DataFrame:
    logger.info(f"Starting evaluation of dataset with {len(dataset)} entries")
    results = []

    for index, row in dataset.iterrows():
        logger.info(f"Evaluating entry {index + 1}/{len(dataset)}")
        start_time = time.time()

        result = evaluate_response(row, num_evaluations)
        results.append(result)

        end_time = time.time()
        logger.info(f"Evaluation for entry {index + 1} completed in {end_time - start_time:.2f} seconds")

    logger.success(f"Evaluation of all {len(dataset)} entries completed")

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Combine original dataset with results
    combined_df = pd.concat([dataset, results_df], axis=1)

    return combined_df


def print_results(results: pd.DataFrame):
    logger.info("Printing evaluation results")

    # Calculate average scores and standard deviations
    avg_scores = {}
    std_dev_scores = {}
    for metric in results['scores'][0].keys():
        if isinstance(results['scores'][0][metric], dict) and 'average_score' in results['scores'][0][metric]:
            scores = [result['average_score'] for result in results['scores'].apply(lambda x: x[metric])]
        elif isinstance(results['scores'][0][metric], (int, float)):
            scores = [result for result in results['scores'].apply(lambda x: x[metric])]
        else:
            logger.warning(f"Skipping non-numeric metric: {metric}")
            continue
        avg_scores[metric] = mean(scores)
        std_dev_scores[metric] = stdev(scores) if len(scores) > 1 else 0

    # Print individual results
    for index, row in results.iterrows():
        logger.info(f"Results for entry {index + 1}:")
        for metric, score in row['scores'].items():
            if isinstance(score, dict) and 'average_score' in score:
                logger.info(f"  {metric} average score: {score['average_score']:.4f}")
            elif isinstance(score, (int, float)):
                logger.info(f"  {metric}: {score:.4f}")
            else:
                logger.info(f"  {metric}: {score}")

        # Log attribution scores
        chunk_attribution_scores = row.get('chunk_attribution_scores', {})
        if chunk_attribution_scores:
            logger.info("Attribution Scores:")
            for chunk, score in chunk_attribution_scores.items():
                logger.info(f"Chunk: '{chunk}")
                logger.info(f"Attribution Score: '{score}")

        logger.info(f"  Metadata: {row['metadata']}")
        logger.info("---")

    # Print average scores and standard deviations
    logger.info("Average scores and standard deviations across all entries:")
    for metric in avg_scores.keys():
        logger.info(f"  {metric}: {avg_scores[metric]:.4f} ± {std_dev_scores[metric]:.4f}")

    # Print overall statistics
    total_time = sum(row['metadata']['total_time'] for _, row in results.iterrows())
    avg_time = total_time / len(results)
    logger.info(f"Total evaluation time: {total_time:.2f} seconds")
    logger.info(f"Average time per entry: {avg_time:.2f} seconds")


if __name__ == "__main__":
    logger.info("Starting combined RAG evaluation")

    # Create a DataFrame with one example
    dataset = pd.DataFrame([{
        "question": "How has China's Belt and Road Initiative affected global geopolitics?",
        "context": "The Belt and Road Initiative (BRI), also known as the One Belt One Road (OBOR) or the Silk Road Economic Belt and the 21st-century Maritime Silk Road, is a global infrastructure development strategy adopted by the Chinese government in 2013 to invest in nearly 150 countries and international organizations. It is considered a centerpiece of Chinese leader Xi Jinping's foreign policy. The BRI forms a central component of Xi's 'Major Country Diplomacy' strategy, which calls for China to assume a greater leadership role for global affairs in accordance with its rising power and status. As of January 2023, 151 countries were listed as having signed up to the BRI. Xi originally announced the strategy as the 'Silk Road Economic Belt' during an official visit to Kazakhstan in September 2013. The initiative was quickly expanded to include the 'Maritime Silk Road', which was announced by Xi in a speech to the Indonesian Parliament in October 2013. BRI projects often include physical infrastructure such as railways, highways, and ports, as well as digital infrastructure like 5G networks. The initiative has been praised for its potential to boost economic growth and connectivity, particularly in developing countries. However, it has also faced criticism, with some observers viewing it as a form of economic imperialism or debt-trap diplomacy.",
        "model_answer": "China's Belt and Road Initiative (BRI) has significantly impacted global geopolitics by expanding China's economic and political influence. It has increased China's global presence through infrastructure investments in numerous countries, particularly in developing regions. This has led to stronger economic ties between China and participating nations, but also raised concerns about debt dependency and China's growing global influence. The initiative has challenged traditional power dynamics, particularly in relation to Western influence in developing countries.",
        "human_answer": "China's Belt and Road Initiative (BRI) has markedly reshaped global geopolitics by expanding China's economic and political reach. It has strengthened China's ties with numerous countries through infrastructure investments, enhancing its global influence. While boosting economic development in participating nations, it has also sparked debates about debt sustainability and China's strategic intentions. The BRI has challenged existing global power structures, particularly Western influence in developing regions, and has become a focal point in discussions about the changing nature of global leadership and economic cooperation. This initiative has also intensified strategic competition between China and other major powers, especially the United States, leading to increased tensions and realignments in international relations.",
        "system_prompt": "You are an AI assistant designed to provide informative and accurate answers based on the given context. Your task is to analyze the provided information, extract relevant details, and formulate a clear, concise, and comprehensive response that directly addresses the question. Ensure your answer is well-structured, balancing depth and brevity. When appropriate, include key points, their relationships, and potential implications. Strive for objectivity and avoid speculation beyond what is supported by the given context."
    }])

    num_evaluations = 5
    logger.info(f"Number of evaluations per metric: {num_evaluations}")

    try:
        start_time = time.time()
        results = evaluate_dataset(dataset, num_evaluations)
        end_time = time.time()

        logger.success("Evaluation of dataset completed")
        logger.info(f"Total evaluation time: {end_time - start_time:.2f} seconds")

        print_results(results)
        results = expand_dict_columns(results)
        results.to_excel("test.xlsx")
    except Exception as e:
        logger.error(f"An error occurred during evaluation: {str(e)}")
        logger.exception("Full traceback:")
