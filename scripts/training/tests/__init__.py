"""
Test suite for the 4D-STEM autoencoder training pipeline.

This package contains comprehensive tests for all components of the refactored
training system, including unit tests, integration tests, and utilities.

Test Structure:
- test_training_config.py: Tests for TrainingConfig dataclass
- test_dataset_manager.py: Tests for DatasetManager class  
- test_model_manager.py: Tests for ModelManager class
- test_trainer_manager.py: Tests for TrainerManager class
- test_training_pipeline.py: Tests for TrainingPipeline orchestrator
- test_integration.py: End-to-end integration tests
- test_utils.py: Test utilities and mock data generators
- conftest.py: Pytest fixtures and configuration

Usage:
    # Run all tests
    pytest scripts/training/tests/
    
    # Run only unit tests (fast)
    pytest scripts/training/tests/ -m unit
    
    # Run only integration tests
    pytest scripts/training/tests/ -m integration
    
    # Run tests with coverage
    pytest scripts/training/tests/ --cov=scripts.training.train
    
    # Run specific test file
    pytest scripts/training/tests/test_training_config.py
"""

__version__ = "1.0.0"