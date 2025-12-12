import yaml
import json
from pathlib import Path
import sys

def load_manifest(path):
    with open(path, "r") as fh:
        return yaml.safe_load(fh)

def assemble_context(manifest):
    ctx = {
        "agent": {
            "name": manifest["agent_name"],
            "version": manifest["version"]
        },
        "capabilities": manifest.get("capabilities", []),
        "context": {}
    }

    for c in manifest.get("context", []):
        if c["type"] == "inline":
            ctx["context"][c["name"]] = c["value"]
        else:
            p = Path(c["source"])
            if p.exists():
                ctx["context"][c["name"]] = [str(x) for x in p.glob("*")]
            else:
                ctx["context"][c["name"]] = []

    return ctx

if __name__ == "__main__":
    file = sys.argv[1] if len(sys.argv) > 1 else "manifest.yaml"
    manifest = load_manifest(file)
    ctx = assemble_context(manifest)
    print(json.dumps(ctx, indent=2))
