"""
Training Package for AlphaGo Baseline

Contains training pipeline for policy network
"""

from .train_pipeline import TrainingPipeline, GoDataset, generate_synthetic_data

__all__ = ['TrainingPipeline', 'GoDataset', 'generate_synthetic_data']
