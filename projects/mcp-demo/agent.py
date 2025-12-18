from mcp_client import MCPClient
from router import RouterLLM
import json

class Agent:
    def __init__(self, manifest_path="manifest.yaml"):
        self.mcp = MCPClient(manifest_path)
        self.context = self.mcp.load_context()
        self.tools = self.mcp.load_tools()
        model_name = self.mcp.get_model_name()
        self.router = RouterLLM(model_name)

    def run(self, query: str):
        tool_names = list(self.tools.keys())

        decision = self.router.route(
            query=query,
            tools=tool_names,
            context=self.context
        )

        tool = decision["tool"]
        args = decision.get("args", {})

        if tool not in self.tools:
            return {"error": f"Tool '{tool}' not registered."}
        
        print(tool)

        tool_functions = self.tools.get(tool, {})
        if "query" in tool_functions:
            function = tool_functions["query"]

        else:
            function = list(self.tools[tool].values())[0]
        
        print(function)
        result = function(**args)

        return {
            "query": query,
            "chosen_tool": tool,
            "arguments": args,
            "result": result
        }
