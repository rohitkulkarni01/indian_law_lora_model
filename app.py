import gradio as gr
import torch
from unsloth import FastLanguageModel
from transformers import TextIteratorStreamer
from threading import Thread

# 1. Load the model from your local GitHub folder
print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "indian_law_lora_model", # Points to the local folder in the repo
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

FastLanguageModel.for_inference(model)

# Custom CSS to remove the stiff Gradio message border/box
custom_css = """
.message-wrap { border: none !important; box-shadow: none !important; }
.message { border: none !important; border-radius: 8px !important; }
"""

def chat_stream(message, history):
    prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert Indian corporate lawyer. Provide accurate, factual advice based strictly on Indian corporate law, SEBI regulations, and the Companies Act, 2013.

Formatting Rules:
* Answer directly in the very first sentence.
* Use clean Markdown headers (##) for sections. NEVER use stacked headers like '### ##'.
* Use flat bullet points (*). NEVER indent bullet points with tabs or 4+ spaces.
* Bold key regulatory bodies (**SEBI**, **RBI**, **MCA**) and specific Acts.
* Keep facts strictly to Indian jurisdiction.<|eot_id|>"""
    
    for item in history:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            user_msg = item[0] if not isinstance(item[0], tuple) else item[0][0]
            assistant_msg = item[1]
            if user_msg: prompt += f"<|start_header_id|>user<|end_header_id|>\n{user_msg}<|eot_id|>"
            if assistant_msg: prompt += f"<|start_header_id|>assistant<|end_header_id|>\n{assistant_msg}<|eot_id|>"
        elif isinstance(item, dict):
            if item.get("role") == "user": prompt += f"<|start_header_id|>user<|end_header_id|>\n{item.get('content', '')}<|eot_id|>"
            elif item.get("role") == "assistant": prompt += f"<|start_header_id|>assistant<|end_header_id|>\n{item.get('content', '')}<|eot_id|>"
                
    prompt += f"<|start_header_id|>user<|end_header_id|>\n{message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    
    dynamic_tokens = min(1200, 500 + (len(message.split()) * 6))
    streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
    
    generate_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=dynamic_tokens,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.12,
        temperature=0.3,
        top_p=0.9,
        do_sample=True
    )
    
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()
    
    partial_text = ""
    for new_token in streamer:
        partial_text += new_token
        yield partial_text

with gr.Blocks(css=custom_css) as demo:
    gr.ChatInterface(
        fn=chat_stream,
        title="⚖️ AI Indian Corporate Lawyer",
        description="Ask anything about Indian Corporate Law"
    )

if __name__ == "__main__":
    demo.launch(share=True, debug=True)