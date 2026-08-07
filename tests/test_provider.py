from types import SimpleNamespace

from app.provider import (
    build_history_input,
    build_response_arguments,
    extract_generated_files,
    extract_web_research,
    normalize_domains,
    supports_context_stuffing,
)


def test_build_history_input() -> None:
    result = build_history_input(
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "system", "content": "Ignored"},
        ]
    )
    assert result == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]


def test_extract_generated_files() -> None:
    raw = {
        "output": [
            {
                "content": [
                    {
                        "annotations": [
                            {
                                "type": "container_file_citation",
                                "container_id": "cntr_1",
                                "file_id": "file_1",
                                "filename": "report.csv",
                            }
                        ]
                    }
                ]
            }
        ]
    }
    response = SimpleNamespace(model_dump=lambda: raw)
    result = extract_generated_files(response)
    assert result == [
        {
            "container_id": "cntr_1",
            "file_id": "file_1",
            "filename": "report.csv",
        }
    ]



def test_zip_is_not_context_stuffed() -> None:
    assert supports_context_stuffing("repository.zip") is False


def test_readme_can_be_context_stuffed() -> None:
    assert supports_context_stuffing("README.md") is True



def test_stream_arguments_request_reasoning_summary() -> None:
    model = SimpleNamespace(deployment="gpt-test")
    arguments = build_response_arguments(
        model=model,
        input_payload="hello",
        instructions="help",
        reasoning_effort="medium",
        verbosity="medium",
        max_output_tokens=1024,
        use_code_interpreter=False,
        provider_file_ids=[],
        use_web_search=False,
        research_depth="quick",
        web_allowed_domains=[],
        web_blocked_domains=[],
        previous_response_id=None,
        stream=True,
    )
    assert arguments["stream"] is True
    assert arguments["reasoning"] == {
        "summary": "auto",
        "effort": "medium",
    }


def test_auto_effort_still_requests_safe_summary() -> None:
    model = SimpleNamespace(deployment="gpt-test")
    arguments = build_response_arguments(
        model=model,
        input_payload="hello",
        instructions="help",
        reasoning_effort="auto",
        verbosity="medium",
        max_output_tokens=1024,
        use_code_interpreter=False,
        provider_file_ids=[],
        use_web_search=False,
        research_depth="quick",
        web_allowed_domains=[],
        web_blocked_domains=[],
        previous_response_id=None,
    )
    assert arguments["reasoning"] == {"summary": "auto"}



def test_web_search_always_uses_thorough_mode() -> None:
    model = SimpleNamespace(deployment="gpt-test")
    arguments = build_response_arguments(
        model=model,
        input_payload="research this",
        instructions="help",
        reasoning_effort="medium",
        verbosity="medium",
        max_output_tokens=1024,
        use_code_interpreter=False,
        provider_file_ids=[],
        use_web_search=True,
        research_depth="quick",
        web_allowed_domains=["https://learn.microsoft.com/docs", "github.com"],
        web_blocked_domains=["pinterest.com"],
        previous_response_id=None,
    )
    assert arguments["tools"] == [
        {
            "type": "web_search",
            "search_context_size": "high",
            "filters": {
                "allowed_domains": ["learn.microsoft.com", "github.com"],
                "blocked_domains": ["pinterest.com"],
            },
        }
    ]
    assert arguments["include"] == [
        "web_search_call.action.sources",
        "web_search_call.results",
    ]


def test_web_search_arguments_thorough_with_code_interpreter() -> None:
    model = SimpleNamespace(deployment="gpt-test")
    arguments = build_response_arguments(
        model=model,
        input_payload="research and calculate",
        instructions="help",
        reasoning_effort="high",
        verbosity="high",
        max_output_tokens=2048,
        use_code_interpreter=True,
        provider_file_ids=["file_1"],
        use_web_search=True,
        research_depth="thorough",
        web_allowed_domains=[],
        web_blocked_domains=[],
        previous_response_id=None,
        stream=True,
    )
    assert arguments["tools"][0]["type"] == "code_interpreter"
    assert arguments["tools"][1] == {
        "type": "web_search",
        "search_context_size": "high",
    }
    assert arguments["include"] == [
        "web_search_call.action.sources",
        "web_search_call.results",
    ]
    assert arguments["stream"] is True


def test_normalize_domains() -> None:
    assert normalize_domains(
        [
            "https://Learn.Microsoft.com/path",
            "learn.microsoft.com",
            " http://github.com/openai ",
            "bad domain",
        ]
    ) == ["learn.microsoft.com", "github.com"]


def test_extract_web_research() -> None:
    raw = {
        "output": [
            {
                "type": "web_search_call",
                "action": {
                    "type": "search",
                    "query": "Azure Responses web search",
                    "sources": [
                        {"type": "url", "url": "https://learn.microsoft.com/a"},
                        {"type": "url", "url": "https://example.com/b"},
                    ],
                },
                "results": [
                    {
                        "url": "https://learn.microsoft.com/a",
                        "title": "Azure documentation",
                        "snippet": "Official documentation snippet",
                    }
                ],
            },
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Azure supports web search.",
                        "annotations": [
                            {
                                "type": "url_citation",
                                "url": "https://learn.microsoft.com/a",
                                "title": "Azure documentation",
                                "start_index": 0,
                                "end_index": 25,
                            }
                        ],
                    }
                ],
            },
        ]
    }
    response = SimpleNamespace(model_dump=lambda: raw)
    research = extract_web_research(response)
    assert research["used"] is True
    assert research["actions"] == [
        {"type": "search", "query": "Azure Responses web search"}
    ]
    assert research["sources"][0]["title"] == "Azure documentation"
    assert research["sources"][0]["snippet"] == "Official documentation snippet"
    assert research["citations"][0]["source_index"] == 1
