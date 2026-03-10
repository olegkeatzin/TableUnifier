"""Тесты для ollama_client.py (с мокированным ollama)."""

from unittest.mock import MagicMock, patch

from table_unifier.config import OllamaConfig
from table_unifier.ollama_client import OllamaClient


class TestOllamaClient:
    @patch("table_unifier.ollama_client.ollama.Client")
    def test_init_default(self, mock_client_cls):
        client = OllamaClient()
        mock_client_cls.assert_called_once_with(host="http://localhost:11434")
        assert client.llm_model == "qwen3.5:9b"
        assert client.embedding_model == "qwen3-embedding:8b"

    @patch("table_unifier.ollama_client.ollama.Client")
    def test_init_custom_config(self, mock_client_cls):
        cfg = OllamaConfig(host="http://remote:11434", llm_model="llama3:8b")
        client = OllamaClient(cfg)
        mock_client_cls.assert_called_once_with(host="http://remote:11434")
        assert client.llm_model == "llama3:8b"

    @patch("table_unifier.ollama_client.ollama.Client")
    def test_generate(self, mock_client_cls):
        mock_instance = MagicMock()
        mock_instance.generate.return_value = {"response": "test output"}
        mock_client_cls.return_value = mock_instance

        client = OllamaClient()
        result = client.generate("hello")

        assert result == "test output"
        mock_instance.generate.assert_called_once_with(
            model="qwen3.5:9b", prompt="hello",
        )

    @patch("table_unifier.ollama_client.ollama.Client")
    def test_generate_custom_model(self, mock_client_cls):
        mock_instance = MagicMock()
        mock_instance.generate.return_value = {"response": "custom"}
        mock_client_cls.return_value = mock_instance

        client = OllamaClient()
        result = client.generate("hi", model="custom:7b")
        assert result == "custom"
        mock_instance.generate.assert_called_once_with(
            model="custom:7b", prompt="hi",
        )

    @patch("table_unifier.ollama_client.ollama.Client")
    def test_embed(self, mock_client_cls):
        mock_instance = MagicMock()
        mock_instance.embed.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
        mock_client_cls.return_value = mock_instance

        client = OllamaClient()
        result = client.embed("test text")

        assert result == [0.1, 0.2, 0.3]
        mock_instance.embed.assert_called_once_with(
            model="qwen3-embedding:8b", input="test text",
        )

    @patch("table_unifier.ollama_client.ollama.Client")
    def test_embed_batch(self, mock_client_cls):
        mock_instance = MagicMock()
        mock_instance.embed.return_value = {
            "embeddings": [[0.1, 0.2], [0.3, 0.4]],
        }
        mock_client_cls.return_value = mock_instance

        client = OllamaClient()
        result = client.embed_batch(["a", "b"])

        assert len(result) == 2
        assert result[0] == [0.1, 0.2]
        assert result[1] == [0.3, 0.4]
