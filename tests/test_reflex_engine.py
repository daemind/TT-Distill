from collections.abc import Generator
from pathlib import Path
from typing import cast
from unittest import mock

import numpy as np
import pytest

from reflex_engine import STABILITY_PROMPT, MultimodalReflexEngine, ReflexEngine


@pytest.fixture
def mock_llama() -> Generator[mock.MagicMock, None, None]:
    with mock.patch("reflex_engine.Llama") as m:
        # Create a mock instance
        mock_instance = mock.MagicMock()
        mock_instance.return_value = {"choices": [{"text": "stability_ok"}]}
        m.return_value = mock_instance
        yield m


def test_reflex_engine_init(mock_llama: mock.MagicMock) -> None:
    engine = ReflexEngine(model_path="fake.gguf", n_ctx=128)
    assert engine.model_path == "fake.gguf"
    mock_llama.assert_called_once()
    args = mock_llama.call_args[1]
    assert args["n_ctx"] == 128
    assert args["model_path"] == "fake.gguf"


def test_reflex_engine_init_with_lora(
    mock_llama: mock.MagicMock, tmp_path: Path
) -> None:
    lora = tmp_path / "lora.bin"
    lora.write_bytes(b"data")
    ReflexEngine(model_path="fake.gguf", lora_path=str(lora))
    args = mock_llama.call_args[1]
    assert args["adapter_path"] == str(lora)
    assert args["adapter_type"] == "lora"


def test_reflex_engine_query(mock_llama: mock.MagicMock) -> None:
    engine = ReflexEngine(model_path="fake.gguf")
    latency, response = engine.query_reflex("test_prompt")

    assert latency > 0
    assert response == "stability_ok"
    assert engine.model is not None
    cast(mock.MagicMock, engine.model).assert_called_with(
        prompt="test_prompt", max_tokens=1, temperature=0.0, stop=None, echo=False
    )


def test_reflex_engine_query_default(mock_llama: mock.MagicMock) -> None:
    engine = ReflexEngine(model_path="fake.gguf")
    _latency, _response = engine.query_reflex()
    cast(mock.MagicMock, engine.model).assert_called_with(
        prompt=STABILITY_PROMPT, max_tokens=1, temperature=0.0, stop=None, echo=False
    )


def test_reflex_engine_with_embeds(mock_llama: mock.MagicMock) -> None:
    engine = ReflexEngine(model_path="fake.gguf")
    embeds = np.random.randn(1, 10, 512).astype(np.float32)
    _latency, response = engine.query_with_embeds(embeds, prompt="base")

    assert response == "stability_ok"
    assert engine.model is not None
    cast(mock.MagicMock, engine.model).assert_called_with(
        prompt="base",
        inputs_embeds=embeds,
        max_tokens=1,
        temperature=0.0,
        stop=None,
        echo=False,
    )


def test_multimodal_reflex_engine_image(
    mock_llama: mock.MagicMock, tmp_path: Path
) -> None:
    # Create a dummy image file
    img_path = tmp_path / "test.jpg"
    img_path.write_bytes(b"dummy_image_data")

    engine = MultimodalReflexEngine(model_path="fake.gguf")
    _latency, response = engine.query_with_image("What's in this image?", str(img_path))

    assert response == "stability_ok"
    # Check that images was passed to the model
    assert engine.model is not None
    call_args = cast(mock.MagicMock, engine.model).call_args[1]
    assert "images" in call_args
    assert len(call_args["images"]) == 1


@mock.patch("subprocess.run")
def test_multimodal_reflex_engine_video(
    mock_run: mock.MagicMock, mock_llama: mock.MagicMock, tmp_path: Path
) -> None:
    # Mock ffmpeg output
    mock_run.return_value.stdout = b"frame_data"

    video_path = tmp_path / "test.mp4"
    video_path.write_text("dummy video")

    engine = MultimodalReflexEngine(model_path="fake.gguf")
    _latency, response = engine.query_with_video(
        "Analyze video", str(video_path), frame_indices=[0, 1]
    )

    assert response == "stability_ok"
    assert mock_run.call_count == 2
    assert engine.model is not None
    call_args = cast(mock.MagicMock, engine.model).call_args[1]
    assert len(call_args["images"]) == 2


def test_reflex_engine_close(mock_llama: mock.MagicMock) -> None:
    engine = ReflexEngine(model_path="fake.gguf")
    engine.close()
    assert not hasattr(engine, "model")


def test_reflex_engine_main(mock_llama: mock.MagicMock) -> None:
    from reflex_engine import main

    with mock.patch("reflex_engine.Node") as mock_node:
        # Mock node iteration
        mock_node.return_value.__iter__.return_value = [
            {"type": "INPUT", "id": "intent_tensor"}
        ]
        with mock.patch("pathlib.Path.exists", return_value=True):
            main()
        mock_node.return_value.send_output.assert_called_once()
