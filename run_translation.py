import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Tencent-Hunyuan/Hy-MT2-1.8B"

print("Đang tải mô hình...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

# Đọc file đầu vào
with open("input.txt", "r", encoding="utf-8") as f:
    text_to_translate = f.read()

# Tạo yêu cầu dịch sang tiếng Việt
prompt = f"Dịch đoạn văn bản sau sang tiếng Việt:\n{text_to_translate}"
inputs = tokenizer(prompt, return_tensors="pt")

print("Đang tiến hành dịch...")
outputs = model.generate(**inputs, max_new_tokens=1024)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Lưu kết quả ra file
with open("output.txt", "w", encoding="utf-8") as f:
    f.write(result)

print("Đã hoàn tất dịch!")
