import os
import torch
import base64
from io import BytesIO
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM

from settings import model_path, device

# 只使用图像功能
os.environ["VISION_ONLY"] = "1"


model = AutoGPTQForCausalLM.from_quantized(
    model_path,
    torch_dtype=torch.bfloat16,
    device=device,
    trust_remote_code=True,
    disable_exllama=True,
    disable_exllamav2=True
)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


# 将 base64 编码的图片转为 PIL.Image
def load_image_b64(b64_data, max_size=None):
    data = base64.b64decode(b64_data) # Bytes
    tmp_buff = BytesIO(data)
    img = Image.open(tmp_buff).convert('RGB')
    tmp_buff.close()
    # 压缩处理
    if max_size is not None:
        width, height = img.size
        max_width = max(width, height)
        if max_width>max_size: # 图片最大宽度为 1500
            ratio = max_size/max_width
            img = img.resize((width*ratio, height*ratio))
    return img


# 使用 图片 聊天 （单轮）
def chat_w_image(question, image):
    msgs = [{'role': 'user', 'content': [image, question]}]

    print(msgs)

    answer = model.chat(
        msgs=msgs,
        tokenizer=tokenizer
    )

    return answer


if __name__ == '__main__':
    import sys
    import readline

    if len(sys.argv)<2:
        print("usage: ochat.py <image-path>")
        sys.exit(2)

    image_path = sys.argv[1]

    image = Image.open(image_path).convert('RGB')

    while True:
        question = input("请输入您的问题：")
        if len(question.strip())==0:
            sys.exit(0)

        print("\n回答：\n", chat_w_image(question, image))

    # 请一步一步思考，先获取图片中所有文字内容，如果其中有英文拼写错误先纠正拼写错误，然后将所有文字翻译为中文，尽可能翻译。翻译的中文前一定写上“译文：”。