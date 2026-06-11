import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class PublicationPlotter:
    """
    Generates high-quality vector assets for academic publications.
    """
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        sns.set_theme(style="whitegrid")
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'figure.titlesize': 18
        })

    def plot_pass_count_vs_accuracy(self, df: pd.DataFrame):
        """
        Plots the progression of Accuracy Metrics relative to Iteration Pass cycles.
        """
        plt.figure(figsize=(7, 5))
        sns.lineplot(data=df, x="ablation_passes", y="f1_score", marker="o", linewidth=2.5, color="#1f77b4")
        plt.title("IMC Accuracy Optimization vs. Iterative Refinement Passes")
        plt.xlabel("Number of Refinement Passes (Iterative Context Loops)")
        plt.ylabel("Downstream Target F1 Score")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "pass_count_vs_accuracy.pdf"), dpi=300)
        plt.close()

    def plot_compression_vs_accuracy(self, df: pd.DataFrame):
        """
        Plots Performance Retention vs Target Compression budgets across alternative methods.
        """
        plt.figure(figsize=(8, 5))
        sns.barplot(data=df, x="method", y="f1_score", hue="token_budget", palette="viridis")
        plt.title("Compression Matrix Performance Retention Analysis")
        plt.xlabel("Compression Methodology Architecture")
        plt.ylabel("Mean Performance Score (F1 Index)")
        plt.legend(title="Token Budget Target")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "compression_vs_accuracy.png"), dpi=300)
        plt.close()