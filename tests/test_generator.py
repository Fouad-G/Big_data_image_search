from pathlib import Path
from recommender.data.dataset import image_generator

def test_image_generator_runs (tmp_path: Path):
    # damit der generator nicht crasht wenn ordner leer ist
    gen= image_generator(tmp_path)
    assert hasattr(gen,"__iter__")

