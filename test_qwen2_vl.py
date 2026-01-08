import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from qwen_vl_utils import process_vision_info
import json
import os

# 检查模型是否已下载
model_path = "/mnt/f/python_work/training/model/Qwen/Qwen2-VL-2B-Instruct"
if not os.path.exists(model_path):
    print(f"模型路径不存在: {model_path}")
    exit(1)

print("正在加载模型...")
try:
    processor = Qwen2VLProcessor.from_pretrained(model_path)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,  # 使用float16减少显存占用
        device_map="auto"
    )
    print("模型加载成功")
except Exception as e:
    print(f"模型加载失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 从示例数据中获取图片路径
demo_json_path = "/mnt/f/python_work/training/Qwen3-VL/qwen-vl-finetune/demo/single_images.json"
if os.path.exists(demo_json_path):
    with open(demo_json_path, 'r', encoding='utf-8') as f:
        demo_data = json.load(f)
    
    if len(demo_data) > 0:
        # 获取第一个示例
        first_item = demo_data[0]

        image_path = os.path.join("/mnt/f/python_work/training/Qwen3-VL/qwen-vl-finetune/", first_item["image"])

        # 检查图片是否存在
        if os.path.exists(image_path):
            print(f"使用示例图片: {image_path}")
            
            # 根据示例数据构建消息
            conversations = first_item["conversations"]
            user_message = next((conv for conv in conversations if conv["from"] == "human"), None)
            
            if user_message:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_path},
                            {"type": "text", "text": user_message["value"]},
                        ],
                    }
                ]
            else:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_path},
                            {"type": "text", "text": "请描述这张图片。"},
                        ],
                    }
                ]
        else:
            print(f"图片不存在: {image_path}")
            # 使用文本测试
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "你好，请介绍一下你自己。"},
                    ],
                }
            ]
    else:
        print("示例数据为空")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "你好，请介绍一下你自己。"},
                ],
            }
        ]
else:
    print("示例JSON文件不存在")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "你好，请介绍一下你自己。"},
            ],
        }
    ]

# 处理输入
try:
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device, torch.float16)

    # 生成输出
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    print("模型输出:", output_text[0])
except Exception as e:
    print(f"推理过程中出现错误: {e}")
    import traceback
    traceback.print_exc()
