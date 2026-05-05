"""Tests for the Ollama client utilities (no LLM calls needed)."""
import pytest
from src.ollama_client import _strip_json_fences, parse_json_response


class TestStripJsonFences:
    def test_strips_json_fence(self):
        raw = "```json\n{\"key\": \"value\"}\n```"
        result = _strip_json_fences(raw)
        assert result == '{"key": "value"}'

    def test_strips_plain_fence(self):
        raw = "```\n{\"key\": \"value\"}\n```"
        result = _strip_json_fences(raw)
        assert result == '{"key": "value"}'

    def test_no_fence_unchanged(self):
        raw = '{"key": "value"}'
        result = _strip_json_fences(raw)
        assert result == raw

    def test_strips_whitespace(self):
        raw = "   \n```json\n[1, 2, 3]\n```\n   "
        result = _strip_json_fences(raw)
        assert result == "[1, 2, 3]"


class TestParseJsonResponse:
    def test_parses_object(self):
        data = parse_json_response('{"clauses": [], "summary": "test"}')
        assert data["clauses"] == []
        assert data["summary"] == "test"

    def test_parses_array(self):
        data = parse_json_response('[{"type": "Liability"}, {"type": "Termination"}]')
        assert len(data) == 2

    def test_parses_with_fence(self):
        raw = '```json\n{"key": "val"}\n```'
        data = parse_json_response(raw)
        assert data["key"] == "val"

    def test_extracts_json_from_prose(self):
        raw = 'Here is the result:\n{"clauses": [{"type": "NDA"}]}\nHope that helps!'
        data = parse_json_response(raw)
        assert "clauses" in data

    def test_invalid_json_raises(self):
        with pytest.raises((ValueError, Exception)):
            parse_json_response("this is not json at all and has no json in it anywhere")
