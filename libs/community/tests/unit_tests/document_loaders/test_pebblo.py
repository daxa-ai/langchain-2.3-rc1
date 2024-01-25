import os
from pathlib import Path

from langchain_core.documents import Document

from langchain_community.document_loaders import CSVLoader, PyPDFLoader

EXAMPLE_DOCS_DIRECTORY = str(Path(__file__).parent.parent.parent / "examples/")


def test_pebblo_import() -> None:
    """Test that the Pebblo safe loader can be imported."""
    from langchain_community.document_loaders import PebbloSafeLoader  # noqa: F401


def test_empty_filebased_loader() -> None:
    """Test basic file based csv loader."""
    # Setup
    from langchain_community.document_loaders import PebbloSafeLoader

    file_path = os.path.join(EXAMPLE_DOCS_DIRECTORY, "test_empty.csv")
    expected_docs: list = []

    # Exercise
    loader = PebbloSafeLoader(
        CSVLoader(file_path=file_path),
        "dummy_app_name",
        "dummy_owner",
        "dummy_description",
    )
    result = loader.load()

    # Assert
    assert result == expected_docs


def test_csv_loader_load_valid_data() -> None:
    # Setup
    from langchain_community.document_loaders import PebbloSafeLoader

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
    loader = PebbloSafeLoader(
        CSVLoader(file_path=file_path),
        "dummy_app_name",
        "dummy_owner",
        "dummy_description",
    )
    result = loader.load()

    # Assert
    assert result == expected_docs


def test_pdf_lazy_load():
    # Setup
    from langchain_community.document_loaders import PebbloSafeLoader

    file_path = os.path.join(
        EXAMPLE_DOCS_DIRECTORY, "multi-page-forms-sample-2-page.pdf"
    )

    # Exercise
    loader = PebbloSafeLoader(
        PyPDFLoader(file_path=file_path),
        "dummy_app_name",
        "dummy_owner",
        "dummy_description",
    )

    result = list(loader.lazy_load())

    # Assert
    assert len(result) == 2
