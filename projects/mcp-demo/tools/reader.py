def read(path: str, num_chars: int = 1500):
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        return {"status": "ok", "preview": text[:num_chars]}
    except Exception as e:
        return {"status": "error", "message": str(e)}
