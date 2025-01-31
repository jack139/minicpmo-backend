import os
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM

os.environ["VISION_ONLY"] = "1"


model = AutoGPTQForCausalLM.from_quantized(
    '../../LLMs/lm_model/MiniCPM-o-2_6-int4',
    torch_dtype=torch.bfloat16,
    device="cuda:0",
    trust_remote_code=True,
    disable_exllama=True,
    disable_exllamav2=True
)

tokenizer = AutoTokenizer.from_pretrained(
    '../../LLMs/lm_model/MiniCPM-o-2_6-int4',
    trust_remote_code=True
)


image = Image.open('data/1-003.jpg').convert('RGB')

question = "OCR to extract text from images"
msgs = [{'role': 'user', 'content': [image, question]}]

answer = model.chat(
    msgs=msgs,
    tokenizer=tokenizer
)
print(answer)
