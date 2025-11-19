#!/usr/bin/env python3
"""
Unit tests for GoogleAdapter using google-genai SDK.

Tests the migration from google-generativeai to google-genai SDK,
validating initialization, configuration, and response handling.
All tests use mocks to avoid real API calls.
"""

import os
import sys
from unittest.mock import Mock, patch, MagicMock
import pytest


@pytest.fixture(autouse=True)
def mock_google_genai():
    """Fixture to mock google.genai module for all tests."""
    # Create mock google module and genai submodule
    mock_google = MagicMock()
    mock_genai = MagicMock()
    mock_types = MagicMock()

    # Set up module structure
    mock_google.genai = mock_genai
    mock_genai.types = mock_types

    # Inject into sys.modules
    with patch.dict(sys.modules, {
        'google': mock_google,
        'google.genai': mock_genai,
        'google.genai.types': mock_types,
    }):
        yield {
            'google': mock_google,
            'genai': mock_genai,
            'types': mock_types,
        }


class TestGoogleAdapterInitialization:
    """Test GoogleAdapter initialization and configuration."""

    def test_google_adapter_initialization(self, mock_google_genai):
        """
        Test that GoogleAdapter initializes correctly with API key and model.

        Validates:
        - Client is created with correct API key
        - Model name is set correctly
        - Default model is gemini-2.0-flash
        """
        # Set up mock client
        mock_client = Mock()
        mock_google_genai['genai'].Client.return_value = mock_client

        # Import after mocks are set up
        from neo.adapters import GoogleAdapter

        # Test with explicit API key
        adapter = GoogleAdapter(api_key="test-api-key")

        # Verify Client was called with correct API key
        mock_google_genai['genai'].Client.assert_called_once_with(api_key="test-api-key")
        assert adapter.api_key == "test-api-key"
        assert adapter.model == "gemini-2.0-flash"
        assert adapter.client == mock_client

    @patch.dict(os.environ, {}, clear=True)
    def test_google_adapter_missing_api_key(self, mock_google_genai):
        """
        Test that GoogleAdapter raises ValueError when no API key is provided.

        Validates:
        - ValueError is raised when API key is missing
        - Error message indicates API key is required
        """
        # Ensure no GOOGLE_API_KEY in environment
        if "GOOGLE_API_KEY" in os.environ:
            del os.environ["GOOGLE_API_KEY"]

        from neo.adapters import GoogleAdapter

        # Should raise ValueError
        with pytest.raises(ValueError, match="Google API key required"):
            GoogleAdapter()

    def test_google_adapter_name(self, mock_google_genai):
        """
        Test that GoogleAdapter.name() returns correct format.

        Validates:
        - name() returns "google/model-name" format
        - Uses the configured model name
        """
        # Set up mock client
        mock_google_genai['genai'].Client.return_value = Mock()

        from neo.adapters import GoogleAdapter

        adapter = GoogleAdapter(
            model="gemini-2.0-flash",
            api_key="test-key"
        )

        assert adapter.name() == "google/gemini-2.0-flash"


class TestGoogleAdapterGenerate:
    """Test GoogleAdapter.generate() method with new SDK."""

    def test_google_adapter_generate(self, mock_google_genai):
        """
        Test that generate() calls the new SDK with correct parameters.

        Validates:
        - Message format conversion (role + parts structure)
        - Config uses types.GenerateContentConfig
        - Correct model, contents, and config passed to generate_content
        """
        # Create mock config
        mock_config_instance = Mock()
        mock_google_genai['types'].GenerateContentConfig.return_value = mock_config_instance

        # Create mock client and response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.text = "Generated response text"
        mock_client.models.generate_content.return_value = mock_response
        mock_google_genai['genai'].Client.return_value = mock_client

        from neo.adapters import GoogleAdapter

        # Create adapter
        adapter = GoogleAdapter(api_key="test-key", model="gemini-2.0-flash")

        # Test messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]

        # Call generate
        result = adapter.generate(
            messages=messages,
            temperature=0.5,
            max_tokens=2048,
            stop=["STOP"],
        )

        # Verify config was created correctly
        mock_google_genai['types'].GenerateContentConfig.assert_called_once_with(
            temperature=0.5,
            max_output_tokens=2048,
            stop_sequences=["STOP"],
        )

        # Verify generate_content was called with correct args
        mock_client.models.generate_content.assert_called_once()
        call_args = mock_client.models.generate_content.call_args

        # Check model parameter
        assert call_args.kwargs["model"] == "gemini-2.0-flash"

        # Check message format conversion
        expected_contents = [
            {"role": "user", "parts": ["You are a helpful assistant."]},
            {"role": "user", "parts": ["Hello!"]},
        ]
        assert call_args.kwargs["contents"] == expected_contents

        # Check config
        assert call_args.kwargs["config"] == mock_config_instance

        # Verify response extraction
        assert result == "Generated response text"

    def test_google_adapter_generate_response_extraction(self, mock_google_genai):
        """
        Test that response.text is correctly extracted from the response.

        Validates:
        - Response text is returned as-is
        - No additional processing or transformation
        """
        # Create mock config
        mock_google_genai['types'].GenerateContentConfig.return_value = Mock()

        # Create mock client and response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.text = "Test response content"
        mock_client.models.generate_content.return_value = mock_response
        mock_google_genai['genai'].Client.return_value = mock_client

        from neo.adapters import GoogleAdapter

        # Create adapter
        adapter = GoogleAdapter(api_key="test-key")

        # Call generate
        result = adapter.generate(
            messages=[{"role": "user", "content": "Test"}],
        )

        # Verify response text extraction
        assert result == "Test response content"
        assert isinstance(result, str)

    def test_google_adapter_generate_response_none(self, mock_google_genai):
        """
        Test that generate() handles None response.text gracefully.

        Validates:
        - ValueError is raised when response.text is None
        - Error message indicates empty response
        """
        # Create mock config
        mock_google_genai['types'].GenerateContentConfig.return_value = Mock()

        # Create mock client and response with None text
        mock_client = Mock()
        mock_response = Mock()
        mock_response.text = None
        mock_client.models.generate_content.return_value = mock_response
        mock_google_genai['genai'].Client.return_value = mock_client

        from neo.adapters import GoogleAdapter

        # Create adapter
        adapter = GoogleAdapter(api_key="test-key")

        # Call generate - should raise ValueError
        with pytest.raises(ValueError, match="API returned empty response"):
            adapter.generate(
                messages=[{"role": "user", "content": "Test"}],
            )

    def test_google_adapter_generate_api_error(self, mock_google_genai):
        """
        Test that generate() handles API errors with clear messages.

        Validates:
        - Invalid API key errors (401/403) are detected
        - Rate limit errors (429) are detected
        - Invalid model errors (404) are detected
        - Network errors are detected
        - Error messages are clear and actionable
        """
        # Create mock config
        mock_google_genai['types'].GenerateContentConfig.return_value = Mock()

        # Create mock client
        mock_client = Mock()
        mock_google_genai['genai'].Client.return_value = mock_client

        from neo.adapters import GoogleAdapter

        # Create adapter
        adapter = GoogleAdapter(api_key="test-key", model="gemini-2.0-flash")

        # Test 401 unauthorized error
        mock_client.models.generate_content.side_effect = Exception("401 Unauthorized")
        with pytest.raises(ValueError, match="Invalid API key"):
            adapter.generate(messages=[{"role": "user", "content": "Test"}])

        # Test 403 forbidden error
        mock_client.models.generate_content.side_effect = Exception("403 Forbidden")
        with pytest.raises(ValueError, match="Invalid API key"):
            adapter.generate(messages=[{"role": "user", "content": "Test"}])

        # Test 429 rate limit error
        mock_client.models.generate_content.side_effect = Exception("429 Rate limit exceeded")
        with pytest.raises(ValueError, match="Rate limit exceeded"):
            adapter.generate(messages=[{"role": "user", "content": "Test"}])

        # Test 404 not found error
        mock_client.models.generate_content.side_effect = Exception("404 Model not found")
        with pytest.raises(ValueError, match="Invalid model 'gemini-2.0-flash'"):
            adapter.generate(messages=[{"role": "user", "content": "Test"}])

        # Test network error
        mock_client.models.generate_content.side_effect = Exception("Network connection failed")
        with pytest.raises(ValueError, match="Network error"):
            adapter.generate(messages=[{"role": "user", "content": "Test"}])
