import gradio as gr
import torch
from transformers import TextIteratorStreamer
from threading import Thread
from unsloth import FastLanguageModel

# 1. Load the model from your local GitHub folder
print("Loading Llama 3.2 1B with local Indian Law LoRA adapters...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "indian_law_lora_model", # Points to the local folder in the repo
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

# Enable native 2x faster inference
FastLanguageModel.for_inference(model)

# 2. UI Styling: Remove Gradio's stiff default message borders
custom_css = """
.message-wrap { border: none !important; box-shadow: none !important; }
.message { border: none !important; border-radius: 8px !important; }
"""

# 3. Setup Stop Tokens (forces the model to stop generating at the exact end of a turn)
eos_ids = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    tokenizer.convert_tokens_to_ids("<|end_of_text|>")
]

# 4. Define the inference streaming function
def chat_stream(message, history):
    # Prompt-agnostic universal formatting rules
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
            u_msg = item[0] if not isinstance(item[0], tuple) else item[0][0]
            a_msg = item[1]
            if u_msg and a_msg:
                # Clean any previous artifacts before passing back into context
                clean_a = a_msg.split("Now, answer this")[0].strip()
                prompt += f"<|start_header_id|>user<|end_header_id|>\n{u_msg}<|eot_id|>"
                prompt += f"<|start_header_id|>assistant<|end_header_id|>\n{clean_a}<|eot_id|>"
        elif isinstance(item, dict):
            role = item.get("role")
            content = item.get("content", "")
            if role == "user":
                prompt += f"<|start_header_id|>user<|end_header_id|>\n{content}<|eot_id|>"
            elif role == "assistant":
                clean_content = content.split("Now, answer this")[0].strip()
                prompt += f"<|start_header_id|>assistant<|end_header_id|>\n{clean_content}<|eot_id|>"

    # Append current turn
    prompt += f"<|start_header_id|>user<|end_header_id|>\n{message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"

    # Send to GPU
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    
    # Dynamic token allocation based on prompt length
    dynamic_tokens = min(1200, 500 + (len(message.split()) * 6))
    
    streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)

    # Generation constraints optimized for local 1B execution
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
        # Prevent streaming of hallucinated self-prompts
        if "Now, answer this question" in partial_text:
            partial_text = partial_text.split("Now, answer this question")[0].rstrip()
            yield partial_text
            break
        yield partial_text

# 5. Build and launch the Gradio UI safely wrapped in Blocks
with gr.Blocks(css=custom_css) as demo:
    gr.ChatInterface(
        fn=chat_stream,
        title="⚖️ AI Indian Corporate Lawyer",
        description="Ask conceptual or statutory questions regarding Indian Corporate Law, Companies Act, 2013, and SEBI regulations."
    )

if __name__ == "__main__":
    # Launching with debug mode enabled to track any backend exceptions easily in Colab/Terminal
    demo.launch(share=True, debug=True)