import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "models/DeepAnalyze-8B"

print(f"正在加载模型: {model_path} ...")
print("这可能需要几分钟，请耐心等待...")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        device_map="auto", 
        trust_remote_code=True
    )
    print("✅ 模型加载成功！")

    prompt = "你好，请介绍一下你自己。"
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    print("正在生成回答...")
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("-" * 20)
    print(f"用户: {prompt}")
    print(f"DeepAnalyze: {response}")
    print("-" * 20)

except Exception as e:
    print(f"❌ 发生错误: {e}")
