import requests
from mcp.server.fastmcp import FastMCP

# ─── Ollama Config ────────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL      = "llama3.1:latest"

mcp = FastMCP("OllamaMCP")

def call_ollama(prompt: str) -> str:
    """Send a prompt to local Ollama and return the response."""
    resp = requests.post(OLLAMA_URL, json={"model": MODEL, "prompt": prompt, "stream": False})
    return resp.json().get("response", "No response")


@mcp.tool()
def ask_ollama(prompt: str) -> str:
    """Ask Ollama a question."""
    return call_ollama(prompt)


@mcp.tool()
def add_numbers(a: int, b: int) -> int:
    """Add two numbers."""
    result = a + b
    print(f"Adding {a} + {b} = {result}")
    return result


@mcp.tool()
def summarize_text(text: str) -> str:
    """Summarize text using Ollama."""
    return call_ollama(f"Summarize this:\n{text}")


if __name__ == "__main__":
    print("\nTesting Tools...\n")
    print("Addition Result:", add_numbers(5, 10))
    print("Ollama Result:", ask_ollama("Explain AI in one line"))
    print("\nMCP Server Running...\n")
    mcp.run()
