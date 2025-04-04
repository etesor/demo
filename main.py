import torch
from transformers import pipeline
import huggingface_hub
import gradio as gr
from dotenv import load_dotenv
import os

load_dotenv()

# Gloabl model initialization
huggingface_hub.login(token=os.getenv("HF_TOKEN"))

device = "cuda" if torch.cuda.is_available() else "cpu"
llama32 = "meta-llama/Llama-3.2-1B-Instruct"
generator = pipeline(model=llama32, device=device, torch_dtype=torch.bfloat16)

def main(prompt):
    prompt = [
        {"role": "system", "content": "You are a helpful assistant, that responds as a cat. Always respond in one sentence."},
        {"role": "user", "content": prompt},
    ]

    generation = generator(
        prompt,
        do_sample=False,
        temperature=1.0,
        top_p=1,
        max_new_tokens=50
    )

    return generation[0]['generated_text'][-1]['content']


if __name__ == "__main__":
    demo = gr.Interface(
        fn=main,
        inputs=["text"],
        outputs=["text"],
    )

    demo.launch()
