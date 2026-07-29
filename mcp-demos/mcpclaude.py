import os
import requests
from datetime import datetime
from mcp.server.fastmcp import FastMCP

# ─── Ollama Config ────────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL      = "llama3.1:latest"   # change to any model you have pulled

def call_ollama(prompt: str) -> str:
    """Send a prompt to local Ollama and return the response."""
    resp = requests.post(OLLAMA_URL, json={"model": MODEL, "prompt": prompt, "stream": False})
    return resp.json().get("response", "No response")

mcp = FastMCP("OllamaMCP")

# =====================================================
# GEMINI CHAT TOOL
# =====================================================
@mcp.tool()
def ask_gemini(prompt: str) -> str:
    """
    Ask Gemini using Vertex AI
    """

    try:

        return call_ollama(prompt)

    except Exception as e:
        return f"Error: {str(e)}"


# =====================================================
# ADDITION TOOL
# =====================================================
@mcp.tool()
def add_numbers(a: int, b: int) -> int:
    """
    Add two numbers
    """

    result = a + b

    print(f"Adding {a} + {b} = {result}")

    return result


# =====================================================
# MULTIPLICATION TOOL
# =====================================================
@mcp.tool()
def multiply_numbers(a: int, b: int) -> int:
    """
    Multiply two numbers
    """

    result = a * b

    print(f"Multiplying {a} * {b} = {result}")

    return result


# =====================================================
# WORD COUNT TOOL
# =====================================================
@mcp.tool()
def count_words(text: str) -> int:
    """
    Count words in text
    """

    return len(text.split())


# =====================================================
# CURRENT TIME TOOL
# =====================================================
@mcp.tool()
def get_current_time() -> str:
    """
    Get current system time
    """

    now = datetime.now()

    return now.strftime("%Y-%m-%d %H:%M:%S")


# =====================================================
# TEXT SUMMARY TOOL
# =====================================================
@mcp.tool()
def summarize_text(text: str) -> str:
    """
    Summarize text using Gemini
    """

    try:

        return call_ollama(f"Summarize this:\n{text}")

    except Exception as e:
        return f"Error: {str(e)}"


# =====================================================
# TRANSLATION TOOL
# =====================================================
@mcp.tool()
def translate_text(text: str, language: str) -> str:
    """
    Translate text into another language
    """

    try:

        prompt_str = f"Translate the following text into {language}.\n\nText:\n{text}"
        return call_ollama(prompt_str)

    except Exception as e:
        return f"Error: {str(e)}"


# =====================================================
# SENTIMENT ANALYSIS TOOL
# =====================================================
@mcp.tool()
def analyze_sentiment(text: str) -> str:
    """
    Analyze sentiment of text
    """

    try:

        return call_ollama(f"Analyze the sentiment of this text. Return ONLY Positive, Negative, or Neutral.\n\nText:\n{text}").strip()

    except Exception as e:
        return f"Error: {str(e)}"


# =====================================================
# CODE EXPLANATION TOOL
# =====================================================
@mcp.tool()
def explain_code(code: str) -> str:
    """
    Explain programming code using Gemini
    """

    try:

        return call_ollama(f"Explain this code in simple terms:\n\n{code}")

    except Exception as e:
        return f"Error: {str(e)}"


# =====================================================
# FILE READER TOOL
# =====================================================
@mcp.tool()
def read_text_file(filepath: str) -> str:
    """
    Read contents of a text file
    """

    try:

        with open(filepath, "r", encoding="utf-8") as file:
            return file.read()

    except Exception as e:
        return f"Error reading file: {str(e)}"


# =====================================================
# AI Q&A TOOL
# =====================================================
@mcp.tool()
def ask_ai(question: str) -> str:
    """
    General AI question answering tool
    """

    try:

        return call_ollama(question)

    except Exception as e:
        return f"Error: {str(e)}"


# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":

    print("\n======================================")
    print(" Vertex AI MCP Server Starting...")
    print("======================================\n")

    # ---------------------------------
    # Test Tools
    # ---------------------------------
    print("Testing MCP Tools...\n")

    print("Addition Test:")
    print(add_numbers(5, 10))

    print("\nMultiplication Test:")
    print(multiply_numbers(4, 6))

    print("\nWord Count Test:")
    print(count_words("Hello world from MCP server"))

    print("\nCurrent Time:")
    print(get_current_time())

    print("\nGemini Test:")
    print(ask_gemini("Explain Artificial Intelligence in one line"))

    print("\nSentiment Test:")
    print(analyze_sentiment("I love this project"))

    print("\n======================================")
    print(" MCP Server Running...")
    print("======================================\n")

    # IMPORTANT FOR CLAUDE DESKTOP
    mcp.run()