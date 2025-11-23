from fastmcp import FastMCP

mcp_server = FastMCP("FastMCP Greeting Server")

@mcp_server.tool
def greet(name: str) -> str:
    """
    Generates a personalized greeting.

    Args:
        name (str): The name of the person to greet.

    Returns:
        str: A greeting message.
    """
    return f"Hello, {name}! Welcome to FastMCP."

if __name__ == "__main__":
    mcp_server.run(
        transport="http",
        port=7777
    )
