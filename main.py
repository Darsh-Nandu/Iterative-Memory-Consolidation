# Open: main.py
import os
import yaml
import pandas as pd
from loguru import logger
from rich.console import Console
from dotenv import load_dotenv

load_dotenv()

from src.models.llm_client import LLMClientManager
from src.chunking.chunker import StructuralChunker
from src.consolidation.engine import IterativeConsolidationEngine
from src.evaluation.evaluator import EvaluationEngine
from src.visualization.plotter import PublicationPlotter
from src.reporting.generator import AcademicReportGenerator
from src.arxiv_fetcher import ArxivDatasetFetcher
import asyncio

async def run_pipeline_async():
    logger.add("logs/experiment.log", rotation="500 MB")
    logger.info("Initializing Experimental Verification Matrix Execution Framework.")
    
    with open("configs/default_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    with open("prompts/memory_prompts.yaml", "r") as f:
        prompts = yaml.safe_load(f)

    # DYNAMIC DATASET ACQUISITION STAGE
    dataset_config = config["dataset"]
    
    if dataset_config["family"].lower() == "arxiv":
        logger.info("Detected arXiv execution configuration setting. Commencing live document generation pipeline...")
        fetcher = ArxivDatasetFetcher(storage_dir="datasets")
        try:
            # Query the web endpoint, fetch the paper, extract textual tokens
            active_document = fetcher.fetch_and_extract(
                search_query=dataset_config.get("search_query", "machine learning"),
                max_results=1
            )
        except Exception as e:
            logger.error(f"Automated arXiv ingestion failed: {e}")
            return
    else:
        # Fallback local folder disk reader if family setup is changed to local assets
        dataset_dir = "datasets"
        txt_files = [f for f in os.listdir(dataset_dir) if f.endswith(".txt")]
        if not txt_files:
            logger.error("No active text datasets available for parsing.")
            return
        with open(os.path.join(dataset_dir, txt_files[0]), "r", encoding="utf-8") as f:
            active_document = f.read()

    # Enforce maximum text length truncation safety buffer to stay within processing limits
    active_document = active_document[:40000]

    # Initialize Client Managers and Processing Framework Engines
    client_manager = LLMClientManager(config)
    chunker = StructuralChunker(
        strategy=config["chunking"]["strategy"],
        size=config["chunking"]["size"],
        overlap=config["chunking"]["overlap"]
    )
    imc_engine = IterativeConsolidationEngine(client_manager, prompts, config)
    evaluator = EvaluationEngine(client_manager, prompts)

    # Convert the long-form text document into chunks
    chunks = chunker.chunk_text(active_document)
    logger.info(f"Successfully split paper source text into {len(chunks)} runtime window segments.")
    
    # Generate verification Question-Answer targets based on the downloaded paper text
    qa_pairs = await evaluator.generate_synthetic_qa(active_document, num_questions=config["evaluation"]["num_questions_per_doc"])

    results_data = []

    # Execute Comparison Matrix Loops over Token Limits and Ablation Profiles
    for budget in config["experiment"]["token_budgets"]:
        logger.info(f"Evaluating Baseline Summary Performance at Target Token Budget: {budget}")
        one_pass_summary = active_document[:int(budget * 4.5)]
        one_pass_res = await evaluator.evaluate_memory_system(one_pass_summary, qa_pairs)
        for r in one_pass_res:
            results_data.append({
                "method": "One-Pass", "token_budget": budget, "ablation_passes": 0,
                "f1_score": r["f1_score"], "semantic_similarity": r["semantic_similarity"]
            })

        for passes in config["experiment"]["ablation_passes"]:
            logger.info(f"Evaluating IMC Framework Performance. Budget Limit: {budget}, Passes: {passes}")
            imc_output = await imc_engine.execute_imc_pipeline(chunks, budget=budget, passes=passes)
            imc_res = await evaluator.evaluate_memory_system(imc_output["final_memory"], qa_pairs)
            
            for r in imc_res:
                results_data.append({
                    "method": "IMC", "token_budget": budget, "ablation_passes": passes,
                    "f1_score": r["f1_score"], "semantic_similarity": r["semantic_similarity"]
                })

    # Save tracking statistics and render data charts
    df = pd.DataFrame(results_data)
    os.makedirs("results/tables", exist_ok=True)
    df.to_csv("results/tables/results.csv", index=False)

    plotter = PublicationPlotter(output_dir="results/figures")
    plotter.plot_pass_count_vs_accuracy(df[df["method"] == "IMC"])
    plotter.plot_compression_vs_accuracy(df)

    report_gen = AcademicReportGenerator(
        results_df=df,
        template_path="experiment_report_template.md",
        output_path="results/reports/experiment_report.md"
    )
    report_gen.generate_report()
    logger.info("Execution complete. Scientific report generated successfully.")

# Main sub-command mapping execution loop
if __name__ == "__main__":
    import typer
    typer.run(asyncio.run(run_pipeline_async()))