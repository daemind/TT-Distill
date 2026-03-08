from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.persistence.embedding_service import EmbeddingService


@pytest.fixture
def embedding_service():
    # Force SENTENCE_TRANSFORMERS_AVAILABLE to False to test fallback or treat as mockable
    return EmbeddingService(use_local_fallback=True)

@pytest.mark.anyio
async def test_embedding_service_local_fallback(embedding_service):
    # Ensure it uses local fallback if sentence_transformers is mocked as unavailable
    with patch("src.persistence.embedding_service.SENTENCE_TRANSFORMERS_AVAILABLE", False):
        embedding_service._available = False
        vec = embedding_service.compute_embedding("test text")
        assert len(vec) == 1024
        assert isinstance(vec[0], float)

        # Test caching
        vec2 = embedding_service.compute_embedding("test text")
        assert vec == vec2

@pytest.mark.anyio
async def test_embedding_service_batch(embedding_service):
    with patch("src.persistence.embedding_service.SENTENCE_TRANSFORMERS_AVAILABLE", False):
        embedding_service._available = False
        texts = ["t1", "t2"]
        vecs = embedding_service.compute_batch_embeddings(texts)
        assert len(vecs) == 2
        assert len(vecs[0]) == 1024

@pytest.mark.anyio
async def test_embedding_service_model_loading_failure():
    service = EmbeddingService(use_local_fallback=False)
    with patch("src.persistence.embedding_service.SENTENCE_TRANSFORMERS_AVAILABLE", True):
        service._available = True
        with patch("src.persistence.embedding_service.SentenceTransformer") as mock_st:
            mock_st.side_effect = Exception("Load failed")
            # If load fails and fallback is False, it should raise or return zero-vec
            # Implementation check: line 130 catches it and returns zero embedding if fallback False
            vec = service.compute_embedding("test")
            assert all(v == 0.0 for v in vec)

@pytest.mark.anyio
async def test_embedding_service_success_model():
    service = EmbeddingService(use_local_fallback=False)
    with patch("src.persistence.embedding_service.SENTENCE_TRANSFORMERS_AVAILABLE", True):
        service._available = True
        with patch("src.persistence.embedding_service.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_st.return_value = mock_model
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])

            # First call loads model
            vec = service.compute_embedding("test")
            assert vec == [0.1, 0.2, 0.3]
            assert service._embedding_dim == 384

            # Second call reuses model
            vec2 = service.compute_embedding("other")
            assert vec2 == [0.1, 0.2, 0.3]
            assert mock_st.call_count == 1

@pytest.mark.anyio
async def test_embedding_service_error_fallback_to_local():
    service = EmbeddingService(use_local_fallback=True)
    with patch("src.persistence.embedding_service.SENTENCE_TRANSFORMERS_AVAILABLE", True):
        service._available = True
        with patch("src.persistence.embedding_service.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_st.return_value = mock_model
            mock_model.get_sentence_embedding_dimension.return_value = 1024
            mock_model.encode.side_effect = Exception("Runtime error")

            # Should fallback to local embedding
            vec = service.compute_embedding("fallback test")
            assert len(vec) == 1024
            assert any(v != 0.0 for v in vec) # Local embedding is non-zero

@pytest.mark.anyio
async def test_embedding_service_get_model_reuse():
    service = EmbeddingService()
    with patch("src.persistence.embedding_service.SENTENCE_TRANSFORMERS_AVAILABLE", True):
        service._available = True
        with patch("src.persistence.embedding_service.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_st.return_value = mock_model
            m1 = service._get_model()
            m2 = service._get_model()
            assert m1 == m2
            assert mock_st.call_count == 1
@pytest.mark.anyio
async def test_embedding_service_get_model_error():
    service = EmbeddingService(use_local_fallback=True)
    with patch("src.persistence.embedding_service.SENTENCE_TRANSFORMERS_AVAILABLE", False):
        service._available = False
        with pytest.raises(RuntimeError, match="sentence_transformers is not installed"):
            service._get_model()

@pytest.mark.anyio
async def test_embedding_service_chunk_key(embedding_service):
    key = embedding_service._generate_chunk_key("some long text", 0, 10)
    assert len(key) == 16

    key2 = embedding_service._generate_chunk_key("some long text")
    assert len(key2) == 16

@pytest.mark.anyio
async def test_embedding_service_clear_cache(embedding_service):
    embedding_service.compute_embedding("test")
    assert len(embedding_service._cache) == 1
    embedding_service.clear_cache()
    assert len(embedding_service._cache) == 0

@pytest.mark.anyio
async def test_embedding_service_set_dim(embedding_service):
    embedding_service.set_embedding_dimension(128)
    vec = embedding_service.compute_embedding("test")
    assert len(vec) == 128
