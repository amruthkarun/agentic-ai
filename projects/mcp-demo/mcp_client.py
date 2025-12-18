import yaml
import json
import importlib
import inspect
from types import FunctionType

class MCPClient:
    def __init__(self, manifest_path="manifest.yaml"):
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.manifest = yaml.safe_load(f)

    def load_context(self):
        ctx = {}
        for entry in self.manifest.get("context", []):
            if entry["type"] == "file":
                with open(entry["path"], "r") as f:
                    ctx[entry["name"]] = json.load(f)
        return ctx

    def load_tools(self):
        tools = {}

        for tool_name, data in self.manifest.get("tools", {}).items():
            module_name = data["module"]

            try:
                module = importlib.import_module(module_name)
            except Exception as e:
                print(f"[ERROR] Could not import module '{module_name}': {e}")
                tools[tool_name] = {}
                continue

            # ONLY load functions defined INSIDE the module (not imported classes!)
            functions = {
                name: fn
                for name, fn in module.__dict__.items()
                if isinstance(fn, FunctionType) and not name.startswith("_")
            }

            if not functions:
                print(f"[WARNING] Tool '{tool_name}' from module '{module_name}' has no usable functions.")

            tools[tool_name] = functions

        return tools

    def get_model_name(self):
        return self.manifest["llm"]["model"]
