import asyncio
import typer
import yaml
import pandas as pd
from loguru import logger
from rich.console import Console
from src.models.llm_client import LLMClientManager
from src.chunking.chunker import StructuralChunker
from src.consolidation.engine import IterativeConsolidationEngine
from src.evaluation.evaluator import EvaluationEngine
from src.visualization.plotter import PublicationPlotter
from src.reporting.generator import AcademicReportGenerator

app = typer.Typer(help="Iterative Memory Consolidation Execution Engine CLI Pipeline Blueprint")
console = Console()

async def run_pipeline_async():
    logger.add("logs/experiment.log", rotation="500 MB")
    logger.info("Initializing Experimental Verification Matrix Execution Framework.")
    
    # Load Configurations
    with open("configs/default_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    with open("prompts/memory_prompts.yaml", "r") as f:
        prompts = yaml.safe_load(f)

    # Mock initialization data for reproduction
    synthetic_document = (
        "Project Sentinel began in 2021 as a secondary cloud routing initiative under Dr. Aris Thorne. "
        "By 2023, the scope transitioned into multi-hop contextual storage frameworks. This change was caused "
        "by structural memory degradation limitations encountered during early testing with legacy transformer loops. "
        "The complete deployment architecture was integrated into production cluster node-omega within the enterprise environment."
    )

    # Initialize Engine Components
    client_manager = LLMClientManager(config)
    chunker = StructuralChunker(
        strategy=config["chunking"]["strategy"],
        size=config["chunking"]["size"],
        overlap=config["chunking"]["overlap"]
    )
    imc_engine = IterativeConsolidationEngine(client_manager, prompts, config)
    evaluator = EvaluationEngine(client_manager, prompts)

    # Run Chunking Pipeline
    chunks = chunker.chunk_text(synthetic_document)
    
    # Generate Synthetic Target Verification Items
    qa_pairs = await evaluator.generate_synthetic_qa(synthetic_document, num_questions=4)

    results_data = []

    # Execution Loop over Budgets and Ablation Targets
    for budget in config["experiment"]["token_budgets"]:
        # Execute One-Pass Baseline Simulation
        logger.info(f"Simulating One-Pass Context Compression Baseline at budget limit: {budget}")
        one_pass_summary = synthetic_document[:int(budget * 0.5)] # Deterministic proxy baseline
        one_pass_res = await evaluator.evaluate_memory_system(one_pass_summary, qa_pairs)
        for r in one_pass_res:
            results_data.append({
                "method": "One-Pass", "token_budget": budget, "ablation_passes": 0,
                "f1_score": r["f1_score"], "semantic_similarity": r["semantic_similarity"]
            })

        # Execute IMC Architecture
        for passes in config["experiment"]["ablation_passes"]:
            logger.info(f"Running IMC Pipeline. Budget: {budget}, Passes: {passes}")
            imc_output = await imc_engine.execute_imc_pipeline(chunks, budget=budget, passes=passes)
            imc_res = await evaluator.evaluate_memory_system(imc_output["final_memory"], qa_pairs)
            
            for r in imc_res:
                results_data.append({
                    "method": "IMC", "token_budget": budget, "ablation_passes": passes,
                    "f1_score": r["f1_score"], "semantic_similarity": r["semantic_similarity"]
                })

    # Save Results & Materialize Reporting Visualizations
    df = pd.DataFrame(results_data)
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
    logger.info("Experimental verification complete. Reports and vector metrics written to disk.")

@app.command()
def run_experiment():
    """Execute the full context compression experiment matrix from the CLI."""
    asyncio.run(run_pipeline_async())

if __name__ == "__main__":
    app()