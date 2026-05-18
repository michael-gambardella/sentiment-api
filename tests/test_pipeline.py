import pytest
from data.pipeline import clean_text, build_dataloaders, MAX_LENGTH


def test_clean_text_removes_html_tags():
    raw = "This was <b>great</b>!<br />"
    assert clean_text(raw) == "This was great!"


def test_clean_text_collapses_whitespace():
    raw = "too   many    spaces\n\nand newlines"
    assert clean_text(raw) == "too many spaces and newlines"


def test_clean_text_empty_string():
    assert clean_text("") == ""


def test_clean_text_no_html():
    text = "A perfectly clean sentence."
    assert clean_text(text) == text


@pytest.fixture(scope="module")
def dataloaders():
    """Build DataLoaders once for all pipeline tests in this module."""
    return build_dataloaders(batch_size=8)


def test_train_loader_batch_shape(dataloaders):
    train_loader, _ = dataloaders
    batch = next(iter(train_loader))

    assert batch["input_ids"].shape == (8, MAX_LENGTH)
    assert batch["attention_mask"].shape == (8, MAX_LENGTH)
    assert batch["labels"].shape == (8,)


def test_test_loader_batch_shape(dataloaders):
    _, test_loader = dataloaders
    batch = next(iter(test_loader))

    assert batch["input_ids"].shape == (8, MAX_LENGTH)
    assert batch["attention_mask"].shape == (8, MAX_LENGTH)


def test_labels_are_binary(dataloaders):
    train_loader, _ = dataloaders
    batch = next(iter(train_loader))
    unique_labels = set(batch["labels"].tolist())

    assert unique_labels.issubset({0, 1})


def test_attention_mask_values_are_binary(dataloaders):
    train_loader, _ = dataloaders
    batch = next(iter(train_loader))
    unique_mask_vals = set(batch["attention_mask"].flatten().tolist())

    assert unique_mask_vals.issubset({0, 1})
