import pandas as pd
import pickle
import pytest
import os
from delay_finder.load_and_save import save_model


@pytest.fixture
def sample_model():
    """Create a sample model for testing."""
    return {'param1': 0.5, 'param2': 'abc'}


def test_save_model(tmp_path, sample_model):
    """Test save_model function."""
    file_path = tmp_path / "test_model.pickle"
    save_model(sample_model, file_path)
    assert os.path.exists(file_path)
    with open(file_path, 'rb') as f:
        loaded_model = pickle.load(f)
    assert loaded_model == sample_model
