from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json

class RouterLLM:
    def __init__(self, model_name="google/flan-t5-small"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def route(self, query: str, tools: list, context: dict):
        """
        Output a JSON decision object:
        {"tool": "<name>", "args": {...}}
        """

        tool_list = ", ".join(tools)

        prompt = f"""
You are a tool-selection router.
Given the user query and available tools, respond *only* in JSON.

Available tools: {tool_list}

Rules:
- If query asks to read/open/extract → tool = "reader"
- If query asks to search/find/look up → tool = "search"
- If query asks to summarize/shorten → tool = "summarize"
- Else → tool = "search"

User query: "{query}"

Respond ONLY in JSON like:
{{"tool": "reader", "args": {{"path": "file.txt"}}}}
"""

        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=200)
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # If model produces invalid JSON, fallback:
        try:
            return json.loads(text)
        except:
            # basic heuristic routing fallback
            if "summar" in query.lower():
                return {"tool": "summarize", "args": {"text": query}}
            if "read" in query.lower():
                return {"tool": "reader", "args": {"path": "sample.txt"}}
            return {"tool": "search", "args": {"term": query}}
