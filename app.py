import os
import chainlit as cl
from ctransformers import AutoModelForCausalLM


# Runs when the chat starts
@cl.on_chat_start
def main():
    # Create the llm
    llm = AutoModelForCausalLM.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
                                               model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
                                               model_type="mistral",
                                               temperature=0.7,
                                               gpu_layers=0,
                                               stream=True,
                                               threads=int(os.cpu_count() / 2),
                                               max_new_tokens=10000)

    # Store the llm in the user session
    cl.user_session.set("llm", llm)


# Runs when a message is sent
@cl.on_message
async def main(message: cl.Message):
    # Retrieve the chain from the user session
    llm = cl.user_session.get("llm")

    msg = cl.Message(
        content="",
    )

    prompt = f"[INST]{message.content}[/INST]"
    for text in llm(prompt=prompt):
        await msg.stream_token(text)

    await msg.send()
