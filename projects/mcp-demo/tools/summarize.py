from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# load once
MODEL_NAME = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

def summarize(text: str, max_words=200):
    prompt = f"Summarize in under {max_words} words:\n{text}"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=80)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"summary": summary}
