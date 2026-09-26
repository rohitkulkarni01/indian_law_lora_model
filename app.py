# 1. Unsloth MUST be imported before transformers/torch
from unsloth import FastLanguageModel
import gradio as gr
import torch
from transformers import TextIteratorStreamer
from threading import Thread

# 2. Load the model from your local GitHub folder
print("Loading Llama 3.2 1B with local Indian Law LoRA adapters...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "indian_law_lora_model",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

FastLanguageModel.for_inference(model)

# 3. UI Styling: Remove Gradio's stiff default message borders
custom_css = """
.message-wrap { border: none !important; box-shadow: none !important; }
.message { border: none !important; border-radius: 8px !important; }
"""

# 4. Stop token IDs to enforce hard stops at the end of each turn
eos_ids = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    tokenizer.convert_tokens_to_ids("<|end_of_text|>")
]

# Helper function to extract plain text regardless of Gradio version formatting
def extract_text(content):
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
            else:
                parts.append(str(part))
        return " ".join(parts)
    elif isinstance(content, dict):
        return content.get("text", str(content))
    return str(content) if content is not None else ""

# 5. Define the streaming chat function
def chat_stream(message, history):
    system_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert Indian corporate lawyer. Answer directly using clean, standard Markdown.
- State the direct answer in the first sentence.
- Use flat bullet points (*) for details. Do not indent bullets.
- Bold key statutory bodies (**SEBI**, **RBI**, **MCA**) and Acts.
- Never output JSON, dictionaries, or prompt instructions.<|eot_id|>"""

    prompt = system_prompt

    # SLIDING WINDOW: Keep only the LAST 1 turn to prevent compounding formatting errors
    recent_history = history[-1:] if len(history) > 0 else []

    for item in recent_history:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            u_raw = item[0] if not isinstance(item[0], tuple) else item[0][0]
            a_raw = item[1]
            u_msg = extract_text(u_raw)
            a_msg = extract_text(a_raw)
            if u_msg and a_msg:
                clean_a = a_msg.split("Now, answer this")[0].strip()
                prompt += f"<|start_header_id|>user<|end_header_id|>\n{u_msg}<|eot_id|>"
                prompt += f"<|start_header_id|>assistant<|end_header_id|>\n{clean_a}<|eot_id|>"
        elif isinstance(item, dict):
            role = item.get("role")
            content_str = extract_text(item.get("content", ""))
            if role == "user":
                prompt += f"<|start_header_id|>user<|end_header_id|>\n{content_str}<|eot_id|>"
            elif role == "assistant":
                clean_content = content_str.split("Now, answer this")[0].strip()
                prompt += f"<|start_header_id|>assistant<|end_header_id|>\n{clean_content}<|eot_id|>"

    # Ensure current user message is converted to clean string
    clean_user_message = extract_text(message)
    prompt += f"<|start_header_id|>user<|end_header_id|>\n{clean_user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"

    # Tokenize and send to GPU
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    dynamic_tokens = min(1200, 500 + (len(clean_user_message.split()) * 6))
    streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)

    generate_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=dynamic_tokens,
        eos_token_id=eos_ids,
        repetition_penalty=1.15,
        temperature=0.2,
        top_p=0.9,
        do_sample=True
    )

    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    partial_text = ""
    for new_token in streamer:
        partial_text += new_token
        if "Now, answer this question" in partial_text:
            partial_text = partial_text.split("Now, answer this question")[0].rstrip()
            yield partial_text
            break
        yield partial_text

# 6. Launch interface with backwards/forwards compatible CSS handling
with gr.Blocks() as demo:
    gr.ChatInterface(
        fn=chat_stream,
        title="⚖️ AI Indian Corporate Lawyer",
        description="Ask conceptual or statutory questions regarding Indian Corporate Law."
    )

if __name__ == "__main__":
    try:
        demo.launch(share=True, debug=True, css=custom_css)
    except TypeError:
        demo.launch(share=True, debug=True)