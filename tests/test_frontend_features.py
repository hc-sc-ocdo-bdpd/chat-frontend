from pathlib import Path

from app.schemas import MessageCreate


def test_tools_are_visible_in_composer() -> None:
    html = Path("app/static/index.html").read_text(encoding="utf-8")
    assert "composer-tool" in html
    assert "> Code<" in html
    assert "Web research" in html
    assert "research-depth" not in html


def test_mathjax_is_configured_for_dynamic_svg_math() -> None:
    html = Path("app/static/index.html").read_text(encoding="utf-8")
    javascript = Path("app/static/app.js").read_text(encoding="utf-8")
    assert "mathjax@4/tex-svg.js" in html
    assert "typeset: false" in html
    assert "scheduleMathTypeset" in javascript
    assert "MathJax.typesetPromise" in javascript
    assert "markdown-math-block" in javascript


def test_web_research_defaults_to_thorough() -> None:
    payload = MessageCreate(
        content="research this",
        model_id="model",
    )
    assert payload.research_depth == "thorough"


def test_frontend_always_sends_thorough_research() -> None:
    javascript = Path("app/static/app.js").read_text(encoding="utf-8")
    assert 'research_depth: "thorough"' in javascript
    assert "el.researchDepth" not in javascript


def test_endpoint_selectors_are_removed() -> None:
    html = Path("app/static/index.html").read_text(encoding="utf-8")
    javascript = Path("app/static/app.js").read_text(encoding="utf-8")
    assert "endpoint-select" not in html
    assert "project-endpoint" not in html
    assert "Default endpoint" not in html
    assert "catalog.endpoints" not in javascript
    assert "el.endpoint" not in javascript
    assert "el.projectEndpoint" not in javascript


def test_frontend_uses_flat_model_catalog() -> None:
    javascript = Path("app/static/app.js").read_text(encoding="utf-8")
    assert "state.catalog.models" in javascript
