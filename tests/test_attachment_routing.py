from app.provider import supports_context_stuffing


def test_archives_are_reserved_for_code_interpreter() -> None:
    for filename in (
        "source.zip",
        "source.tar",
        "source.tar.gz",
        "compiled.bin",
    ):
        assert supports_context_stuffing(filename) is False


def test_documents_can_be_direct_model_inputs() -> None:
    for filename in (
        "README.md",
        "report.pdf",
        "source.py",
        "config.yaml",
        "table.csv",
    ):
        assert supports_context_stuffing(filename) is True
