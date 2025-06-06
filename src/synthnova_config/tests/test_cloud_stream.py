import pytest
from pydantic import ValidationError
from synthnova_config.cloud.cloud_stream import CloudStreamingConfig


def test_default_config():
    """Test default configuration initialization"""
    config = CloudStreamingConfig()
    assert config.webrtc_cloud_streaming is False
    assert config.webrtc_host == "0.0.0.0"
    assert config.webrtc_port == 8080
    assert config.nvidia_native_cloud_streaming is False
    assert config.nvidia_webrtc_cloud_streaming is False


def test_custom_config():
    """Test custom configuration initialization"""
    config = CloudStreamingConfig(
        webrtc_cloud_streaming=True,
        webrtc_host="127.0.0.1",
        webrtc_port=9000,
        nvidia_native_cloud_streaming=True,
        nvidia_webrtc_cloud_streaming=False,
    )
    assert config.webrtc_cloud_streaming is True
    assert config.webrtc_host == "127.0.0.1"
    assert config.webrtc_port == 9000
    assert config.nvidia_native_cloud_streaming is True
    assert config.nvidia_webrtc_cloud_streaming is False


def test_invalid_port():
    """Test validation of invalid port numbers"""
    with pytest.raises(ValidationError):
        CloudStreamingConfig(webrtc_port=80)  # Port < 1024

    with pytest.raises(ValidationError):
        CloudStreamingConfig(webrtc_port=70000)  # Port > 65535


def test_to_dict():
    """Test conversion to dictionary"""
    config = CloudStreamingConfig(
        webrtc_cloud_streaming=True, webrtc_host="127.0.0.1", webrtc_port=8080
    )
    config_dict = config.to_dict()
    assert isinstance(config_dict, dict)
    assert config_dict["webrtc_cloud_streaming"] is True
    assert config_dict["webrtc_host"] == "127.0.0.1"
    assert config_dict["webrtc_port"] == 8080
    assert config_dict["nvidia_native_cloud_streaming"] is False
    assert config_dict["nvidia_webrtc_cloud_streaming"] is False


def test_from_dict():
    """Test creation from dictionary"""
    data = {
        "webrtc_cloud_streaming": True,
        "webrtc_host": "127.0.0.1",
        "webrtc_port": 8080,
        "nvidia_native_cloud_streaming": True,
        "nvidia_webrtc_cloud_streaming": False,
    }
    config = CloudStreamingConfig.from_dict(data)
    assert config.webrtc_cloud_streaming is True
    assert config.webrtc_host == "127.0.0.1"
    assert config.webrtc_port == 8080
    assert config.nvidia_native_cloud_streaming is True
    assert config.nvidia_webrtc_cloud_streaming is False


def test_invalid_dict():
    """Test creation from invalid dictionary"""
    invalid_data = {
        "webrtc_cloud_streaming": "not_a_boolean",  # Invalid type
        "webrtc_port": "not_a_number",  # Invalid type
    }
    with pytest.raises(ValidationError):
        CloudStreamingConfig.from_dict(invalid_data)


def test_extra_fields():
    """Test that extra fields are forbidden"""
    with pytest.raises(ValidationError):
        CloudStreamingConfig(extra_field="value")
