"""
Nomic Embed Text - Ollama Embedding Client
==========================================
This script generates vector embeddings for text using the 'nomic-embed-text' model
hosted locally or on any remote Ollama server.

Features:
- Configurable server URL (via CLI flag --url, environment variable OLLAMA_BASE_URL, or code parameter)
- Supports single strings or lists/batches of strings
- Supports both official `ollama.Client` and standard HTTP `requests` (/api/embed)
- Handles Nomic task prefixes ('search_document: ' and 'search_query: ') for optimal retrieval performance
"""

import os
import sys
import argparse
from typing import List, Union
import requests
import ollama

DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = "nomic-embed-text"


class OllamaEmbeddingClient:
    """Client for generating embeddings via an Ollama instance."""

    def __init__(self, base_url: str = DEFAULT_OLLAMA_URL, model: str = DEFAULT_MODEL):
        """
        Initialize the embedding client.

        :param base_url: Base URL of the Ollama server (e.g., 'http://localhost:11434'
                         or 'http://<server-ip>:11434')
        :param model: Ollama embedding model name (default: 'nomic-embed-text')
        """
        # Normalize base URL (strip trailing slash)
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = ollama.Client(host=self.base_url)

    def get_embeddings_sdk(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        Generate embeddings using the official `ollama` Python SDK.

        :param texts: Single string or list of strings to embed.
        :return: List of embedding vectors (each vector is a list of floats, dimension 768).
        """
        input_data = [texts] if isinstance(texts, str) else texts
        response = self._client.embed(model=self.model, input=input_data)
        return response.embeddings

    def get_embeddings_http(self, texts: Union[str, List[str]], timeout: int = 60) -> List[List[float]]:
        """
        Generate embeddings using direct HTTP POST requests to `/api/embed`.
        Useful for servers or microservices where you prefer standard REST requests.

        :param texts: Single string or list of strings to embed.
        :param timeout: HTTP request timeout in seconds.
        :return: List of embedding vectors (dimension 768).
        """
        endpoint = f"{self.base_url}/api/embed"
        input_data = [texts] if isinstance(texts, str) else texts

        payload = {
            "model": self.model,
            "input": input_data
        }

        response = requests.post(endpoint, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        return data.get("embeddings", [])

    def health_check(self) -> bool:
        """Check if the Ollama server is reachable and active."""
        try:
            res = requests.get(self.base_url, timeout=5)
            return res.status_code == 200
        except Exception as e:
            print(f"[!] Health check failed for {self.base_url}: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate text embeddings using nomic-embed-text via Ollama"
    )
    parser.add_argument(
        "--url", "-u",
        default=DEFAULT_OLLAMA_URL,
        help=f"Ollama server URL (default: {DEFAULT_OLLAMA_URL} or OLLAMA_BASE_URL env var)"
    )
    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_MODEL,
        help=f"Model name (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--text", "-t",
        nargs="+",
        help="One or more text strings to embed. If not provided, a default demo batch is used."
    )
    parser.add_argument(
        "--method",
        choices=["sdk", "http"],
        default="sdk",
        help="API invocation method: 'sdk' (ollama library) or 'http' (direct REST requests)"
    )

    args = parser.parse_args()

    # Sample texts if none provided
    sample_texts = args.text or [
        "search_document: Ollama is an open-source tool that allows users to run LLMs locally.",
        "search_document: nomic-embed-text is a high-performing open-source embedding model with 8192 context window.",
        "search_query: What embedding models can I run locally with Ollama?"
    ]

    print(f"==================================================")
    print(f" Connecting to Ollama Server: {args.url}")
    print(f" Model                      : {args.model}")
    print(f" Method                     : {args.method.upper()}")
    print(f" Input Texts Count          : {len(sample_texts)}")
    print(f"==================================================")

    client = OllamaEmbeddingClient(base_url=args.url, model=args.model)

    print("\n1. Checking Server Connectivity...")
    if not client.health_check():
        print(f"[-] Could not connect to Ollama at {args.url}.")
        print("    Ensure the server is running.")
        print("    If hosting remotely, ensure OLLAMA_HOST=0.0.0.0:11434 is set on the server.")
        sys.exit(1)
    print("    [+] Server is up and reachable!\n")

    print("2. Generating Embeddings...")
    try:
        if args.method == "sdk":
            embeddings = client.get_embeddings_sdk(sample_texts)
        else:
            embeddings = client.get_embeddings_http(sample_texts)

        print(f"    [+] Successfully generated {len(embeddings)} embedding vector(s).\n")

        for idx, (txt, vec) in enumerate(zip(sample_texts, embeddings)):
            print(f"--- Text {idx + 1} ---")
            print(f"Content    : {txt[:70]}{'...' if len(txt) > 70 else ''}")
            print(f"Vector Dim : {len(vec)}")
            print(f"Sample Vec : [{vec[0]:.5f}, {vec[1]:.5f}, {vec[2]:.5f}, ..., {vec[-1]:.5f}]")
            print()

        print("==================================================")
        print("SUCCESS! Embeddings generated successfully.")
        print(f"To use a remote server, specify:")
        print(f"  python embed_client.py --url http://<YOUR_SERVER_IP>:11434")
        print(f"  or set OLLAMA_BASE_URL=http://<YOUR_SERVER_IP>:11434")
        print("==================================================")

    except Exception as e:
        print(f"[-] Error generating embeddings: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
