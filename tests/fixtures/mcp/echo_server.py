"""Minimal stdio MCP server used by the integration test."""

from mcp.server.fastmcp import FastMCP

server = FastMCP("echo")


@server.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@server.tool()
def shout(text: str) -> str:
    """Upper-case the text."""
    return text.upper()


if __name__ == "__main__":
    server.run()
