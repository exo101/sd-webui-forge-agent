# =============================================================================
# Canva MCP Integration — Connects Canva's remote MCP server to the Agent
#
# Features:
#   - OAuth 2.0 authentication (DCR + token persistence)
#   - Local HTTP callback server for OAuth redirect (port 8765)
#   - Persistent background asyncio event loop for async MCP operations
#   - Dynamic tool discovery & registration into Agent tool registry
#   - Tool call forwarding from Agent to Canva MCP
# =============================================================================

import os
import json
import threading
import asyncio
import webbrowser
from pathlib import Path
from typing import Optional, Dict, Any, List

# Extension root directory
EXT_DIR = Path(__file__).parent.parent
TOKENS_PATH = EXT_DIR / "canva_tokens.json"

# Canva MCP server endpoint
CANVA_MCP_URL = "https://mcp.canva.com/mcp"

# Local OAuth callback server
CALLBACK_HOST = "127.0.0.1"
CALLBACK_PORT = 8765
CALLBACK_PATH = "/callback"
REDIRECT_URI = f"http://{CALLBACK_HOST}:{CALLBACK_PORT}{CALLBACK_PATH}"

# Background event loop & session state
_loop: Optional[asyncio.AbstractEventLoop] = None
_loop_thread: Optional[threading.Thread] = None
_oauth_callback_server = None
_oauth_auth_event = threading.Event()
_oauth_auth_code: Optional[str] = None
_oauth_auth_state: Optional[str] = None
_canva_tools: List[dict] = []  # Cached Canva MCP tools (OpenAI schema format)


# =============================================================================
# Token Storage (file-based)
# =============================================================================

class FileTokenStorage:
    """File-based token storage implementing MCP's TokenStorage protocol."""

    def __init__(self, path: Path):
        self.path = path

    async def get_tokens(self):
        """Get stored OAuth tokens."""
        from mcp.client.auth import OAuthToken
        try:
            if self.path.exists():
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                tok = data.get("tokens")
                if tok:
                    return OAuthToken(**tok)
        except Exception as e:
            print(f"[Canva] Token storage read error: {e}")
        return None

    async def set_tokens(self, tokens) -> None:
        """Store OAuth tokens."""
        try:
            data = {}
            if self.path.exists():
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            data["tokens"] = tokens.model_dump()
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[Canva] Token storage write error: {e}")

    async def get_client_info(self):
        """Get stored client registration info."""
        from mcp.client.auth import OAuthClientInformationFull
        try:
            if self.path.exists():
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            info = data.get("client_info")
            if info:
                return OAuthClientInformationFull(**info)
        except Exception as e:
            print(f"[Canva] Client info read error: {e}")
        return None

    async def set_client_info(self, client_info) -> None:
        """Store client registration info."""
        try:
            data = {}
            if self.path.exists():
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            data["client_info"] = client_info.model_dump()
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[Canva] Client info write error: {e}")


# =============================================================================
# OAuth Callback HTTP Server (runs in background loop)
# =============================================================================

def _start_callback_server():
    """Start a lightweight HTTP server to capture OAuth redirect."""
    global _oauth_callback_server, CALLBACK_PORT, REDIRECT_URI

    from http.server import HTTPServer, BaseHTTPRequestHandler

    class _OAuthHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            global _oauth_auth_code, _oauth_auth_state
            parsed = self.path.split("?")
            if len(parsed) < 2 or parsed[0] != CALLBACK_PATH:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"Not found")
                return

            params = {}
            for part in parsed[1].split("&"):
                if "=" in part:
                    k, v = part.split("=", 1)
                    params[k] = v

            if "code" in params:
                _oauth_auth_code = params["code"]
                _oauth_auth_state = params.get("state")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                html = (
                    "<html><body>"
                    "<h2>✅ Canva 授权成功！</h2>"
                    "<p>可以关闭此页面，返回 WebUI 继续使用 Canva 功能。</p>"
                    "<script>setTimeout(function(){window.close();},3000);</script>"
                    "</body></html>"
                )
                self.wfile.write(html.encode("utf-8"))
                _oauth_auth_event.set()
            elif "error" in params:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(f"OAuth error: {params['error']}".encode())
                _oauth_auth_code = None
                _oauth_auth_event.set()
            else:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Missing code parameter")

        def log_message(self, format, *args):
            pass  # Suppress logs

    try:
        _oauth_callback_server = HTTPServer((CALLBACK_HOST, CALLBACK_PORT), _OAuthHandler)
        server_thread = threading.Thread(target=_oauth_callback_server.serve_forever, daemon=True)
        server_thread.start()
        print(f"[Canva] OAuth callback server started on {REDIRECT_URI}")
    except OSError as e:
        print(f"[Canva] Warning: Could not start callback server on port {CALLBACK_PORT}: {e}")
        print(f"[Canva] Will try port {CALLBACK_PORT + 1}...")
        # Try next port
        CALLBACK_PORT += 1
        REDIRECT_URI = f"http://{CALLBACK_HOST}:{CALLBACK_PORT}{CALLBACK_PATH}"
        try:
            _oauth_callback_server = HTTPServer((CALLBACK_HOST, CALLBACK_PORT), _OAuthHandler)
            server_thread = threading.Thread(target=_oauth_callback_server.serve_forever, daemon=True)
            server_thread.start()
            print(f"[Canva] OAuth callback server started on {REDIRECT_URI}")
        except Exception as e2:
            print(f"[Canva] Failed to start callback server: {e2}")


# =============================================================================
# Background Event Loop Management
# =============================================================================

def _ensure_loop():
    """Ensure a persistent background asyncio event loop is running."""
    global _loop, _loop_thread
    if _loop is not None and _loop.is_running():
        return

    def _run_loop():
        global _loop
        _loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop)
        _loop.run_forever()

    _loop_thread = threading.Thread(target=_run_loop, daemon=True)
    _loop_thread.start()

    # Wait for loop to be ready
    import time
    for _ in range(50):
        if _loop is not None and _loop.is_running():
            break
        time.sleep(0.1)

    print(f"[Canva] Background asyncio loop started")


def run_async(coro, timeout=320):
    """Run an async coroutine on the background loop and block until done.
    Default timeout 320s to accommodate OAuth flow (300s user auth wait + buffer)."""
    _ensure_loop()
    future = asyncio.run_coroutine_threadsafe(coro, _loop)
    return future.result(timeout=timeout)


# =============================================================================
# MCP Connection & Tool Management
# =============================================================================

def _make_oauth_provider():
    """Create an OAuthClientProvider with file-based token storage."""
    from mcp.client.auth import OAuthClientProvider, OAuthClientMetadata

    storage = FileTokenStorage(TOKENS_PATH)

    # Client metadata for DCR (Dynamic Client Registration)
    client_metadata = OAuthClientMetadata(
        redirect_uris=[REDIRECT_URI],
        token_endpoint_auth_method="none",
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        client_name="SD Webui Agent",
    )

    # Handlers for OAuth flow
    async def redirect_handler(auth_url: str) -> None:
        """Open browser to Canva authorization page."""
        print(f"[Canva] Opening auth URL: {auth_url}")
        try:
            webbrowser.open(auth_url)
        except Exception as e:
            print(f"[Canva] Could not open browser: {e}")
            print(f"[Canva] Please manually open: {auth_url}")

    async def callback_handler():
        """Wait for OAuth redirect callback, return (auth_code, state)."""
        global _oauth_auth_code, _oauth_auth_state
        _oauth_auth_event.clear()
        _oauth_auth_code = None
        _oauth_auth_state = None
        # Wait up to 5 minutes for user to authenticate
        if _oauth_auth_event.wait(timeout=300):
            return _oauth_auth_code, _oauth_auth_state
        raise RuntimeError("OAuth authentication timed out (5 minutes)")

    return OAuthClientProvider(
        server_url=CANVA_MCP_URL,
        client_metadata=client_metadata,
        storage=storage,
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
    )


async def _discover_tools():
    """Connect to Canva MCP, authenticate if needed, and discover tools.
    Uses a temporary connection (tokens are cached for future calls).
    """
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    oauth_provider = _make_oauth_provider()

    async with streamablehttp_client(
        url=CANVA_MCP_URL,
        auth=oauth_provider,
        timeout=60,
    ) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            print("[Canva] MCP session initialized successfully")

            # Discover tools
            tools_result = await session.list_tools()
            _canva_tools.clear()
            for tool in tools_result.tools:
                # Convert MCP tool schema to OpenAI function-calling format
                openai_tool = {
                    "type": "function",
                    "function": {
                        "name": f"canva_{tool.name}",
                        "description": tool.description or f"Canva tool: {tool.name}",
                        "parameters": tool.inputSchema or {"type": "object", "properties": {}},
                    },
                    "_canva_name": tool.name,  # Original Canva tool name
                }
                _canva_tools.append(openai_tool)

            print(f"[Canva] Discovered {len(_canva_tools)} Canva tools")
            for t in _canva_tools:
                print(f"  - {t['function']['name']}")

            return True


async def _call_tool_async(canva_tool_name: str, args: dict):
    """Connect to Canva MCP and call a tool (async)."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    oauth_provider = _make_oauth_provider()

    async with streamablehttp_client(
        url=CANVA_MCP_URL,
        auth=oauth_provider,
        timeout=60,
    ) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(canva_tool_name, args)

            if hasattr(result, "isError") and result.isError:
                error_text = ""
                if result.content:
                    for block in result.content:
                        if hasattr(block, "text"):
                            error_text += block.text
                return {"status": "error", "message": error_text or "Canva tool error"}

            # Extract text content
            texts = []
            images_base64 = []
            if result.content:
                for block in result.content:
                    if hasattr(block, "text"):
                        texts.append(block.text)
                    elif hasattr(block, "data"):
                        images_base64.append(block.data)

            response = {"status": "success", "data": "\n".join(texts) if texts else "Tool executed"}
            if images_base64:
                response["images"] = images_base64
            return response


def connect_canva() -> dict:
    """Synchronous entry point to connect to Canva MCP.
    Starts OAuth flow if needed. Returns status dict.
    """
    print("[Canva] connect_canva() called — starting OAuth flow...")
    # If already have tools discovered, return success
    if is_connected():
        return {
            "status": "already_connected",
            "message": f"Canva 已连接，可用工具数: {len(_canva_tools)}",
            "tools": [t["function"]["name"] for t in _canva_tools],
        }

    # Start callback server if not running
    _start_callback_server()

    try:
        _ensure_loop()
        run_async(_discover_tools())
        return {
            "status": "connected",
            "message": f"Canva 连接成功！可用工具数: {len(_canva_tools)}",
            "tools": [t["function"]["name"] for t in _canva_tools],
        }
    except Exception as e:
        import traceback
        print(f"[Canva] Connection failed: {traceback.format_exc()}")
        return {
            "status": "error",
            "message": f"Canva 连接失败: {str(e)}",
            "hint": "请确保已在 Canva 注册并授权，检查网络连接",
        }


def is_connected() -> bool:
    """Check if Canva tools have been discovered (connected & authenticated)."""
    return len(_canva_tools) > 0


def get_tools() -> List[dict]:
    """Get Canva tools in OpenAI schema format (without private fields)."""
    return [
        {k: v for k, v in t.items() if not k.startswith("_")}
        for t in _canva_tools
    ]


def get_tool_callable(canva_tool_name: str):
    """Get a sync callable that forwards to the Canva MCP session."""
    def _callable(**kwargs):
        return call_canva_tool(canva_tool_name, kwargs)
    return _callable


def call_canva_tool(canva_tool_name: str, args: dict) -> dict:
    """Forward a tool call to Canva MCP (synchronous wrapper).
    Creates a fresh connection each time (tokens cached automatically).
    """
    if not is_connected():
        return {
            "status": "not_connected",
            "message": "Canva 未连接，请先调用 canva_connect 工具进行授权连接",
        }

    try:
        _ensure_loop()
        return run_async(_call_tool_async(canva_tool_name, args))
    except Exception as e:
        import traceback
        print(f"[Canva] Tool call error: {traceback.format_exc()}")
        return {"status": "error", "message": str(e)}


def try_auto_connect():
    """Try to auto-connect using stored tokens (no browser prompt).
    Returns True if connected, False if user auth needed.
    Only attempts auto-connect if valid tokens exist; never triggers full OAuth here."""
    # First, validate token file integrity before doing anything
    if not TOKENS_PATH.exists():
        return False
    try:
        with open(TOKENS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Must have at least tokens or client_info to attempt auto-connect
        if not data or not isinstance(data, dict):
            raise ValueError("Token file is empty or invalid")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"[Canva] Token file corrupted ({e}), removing it")
        try:
            TOKENS_PATH.unlink()
        except Exception:
            pass
        return False

    try:
        print("[Canva] Found stored tokens, attempting auto-connect...")
        _ensure_loop()
        # Use a short timeout for auto-connect to avoid blocking startup
        run_async(_discover_tools(), timeout=30)
        print("[Canva] Auto-connect successful")
        return True
    except Exception as e:
        print(f"[Canva] Auto-connect failed (user auth may be needed): {e}")
        # Clear stale/invalid tokens
        try:
            TOKENS_PATH.unlink()
        except Exception:
            pass
    return False
