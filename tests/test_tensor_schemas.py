"""
Unit Tests for Multimodal Tensor Schema Validation per Specification V8.0
"""

import pytest
import torch
from stock_rl.data.synthetic_data import generate_synthetic_batch
from stock_rl.data.tensor_validator import validate_multimodal_batch, TensorSchemaError


def test_synthetic_batch_schema():
    batch = generate_synthetic_batch(batch_size=16, seed=42)
    assert validate_multimodal_batch(batch) is True


def test_schema_validator_catches_invalid_dimension():
    batch = generate_synthetic_batch(batch_size=8, seed=42)
    # Tamper with technical feature dimension (e.g. 15 instead of 16)
    batch.technical = batch.technical[:, :, :15]

    with pytest.raises(TensorSchemaError, match="Shape mismatch for technical"):
        validate_multimodal_batch(batch)


def test_schema_validator_catches_nan():
    batch = generate_synthetic_batch(batch_size=8, seed=42)
    # Inject NaN into price tensor
    batch.price[0, 0, 0] = float("nan")

    with pytest.raises(TensorSchemaError, match="NaN detected in tensor price"):
        validate_multimodal_batch(batch)


def test_schema_validator_catches_inf():
    batch = generate_synthetic_batch(batch_size=8, seed=42)
    # Inject Inf into macro tensor
    batch.macro[0, 0, 0] = float("inf")

    with pytest.raises(TensorSchemaError, match="Inf detected in tensor macro"):
        validate_multimodal_batch(batch)
