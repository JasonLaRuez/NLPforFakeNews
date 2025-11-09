"""
PreProcessingDataset package
============================

A lightweight preprocessing toolkit for fake-news datasets.
Provides text deduplication (exact and near-duplicate removal)
and stratified train/validation/test splitting.

Usage:
    from PreProcessingDataset import PreprocessData
"""

from .PreProcessingDataset import PreprocessDataOneDF

__all__ = ["PreprocessData"]
