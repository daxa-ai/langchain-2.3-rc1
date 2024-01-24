import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import CSVLoader

EXAMPLE_DOCS_DIRECTORY = str(Path(__file__).parent.parent / "examples/")

def test_daxa_import() -> None:
    """Test that the Daxa safe loader can be imported."""
    from langchain_community.document_loaders import DaxaSafeLoader  # noqa: F401

def test_empty_filebased_loader() -> None:
    """Test basic file based csv loader."""
    # Setup
    from langchain_community.document_loaders import DaxaSafeLoader
    file_path = os.path.join(EXAMPLE_DOCS_DIRECTORY, "test_empty.csv")
    expected_docs: list = []

    # Exercise
    loader = DaxaSafeLoader(
        CSVLoader(file_path=file_path),
        "dummy_app_name", "dummy_owner","dummy_description"
        )
    result = loader.load()

    # Assert
    assert result == expected_docs

def test_csv_loader_load_valid_data() -> None:
    # Setup
    from langchain_community.document_loaders import DaxaSafeLoader
    file_path = os.path.join(EXAMPLE_DOCS_DIRECTORY, "test_nominal.csv")
    expected_docs = [
        Document(
            page_content="column1: value1\ncolumn2: value2\ncolumn3: value3",
            metadata={"source": file_path, "row": 0},
        ),
        Document(
            page_content="column1: value4\ncolumn2: value5\ncolumn3: value6",
            metadata={"source": file_path, "row": 1},
        ),
    ]

    # Exercise
    loader = DaxaSafeLoader(
        CSVLoader(file_path=file_path),
        "dummy_app_name", "dummy_owner","dummy_description"
        )
    result = loader.load()

    # Assert
    assert result == expected_docs

def test_csv_lazy_load():
        # Setup
    from langchain_community.document_loaders import DaxaSafeLoader
    file_path = os.path.join(EXAMPLE_DOCS_DIRECTORY, "test_nominal.csv")
    expected_docs = [
        Document(
            page_content="column1: value1\ncolumn2: value2\ncolumn3: value3",
            metadata={"source": file_path, "row": 0},
        ),
        Document(
            page_content="column1: value4\ncolumn2: value5\ncolumn3: value6",
            metadata={"source": file_path, "row": 1},
        ),
    ]

    # Exercise
    loader = DaxaSafeLoader(
        CSVLoader(file_path=file_path),
        "dummy_app_name", "dummy_owner","dummy_description"
        )

    result = []
    for doc in loader.lazy_load():
        result.extend(doc)

    # Assert
    assert result == expected_docs
