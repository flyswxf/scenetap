from transformers import CLIPTokenizer
# 这一步会自动把必要的文件下载到当前目录的 clip-vit-base-patch32 文件夹中
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
tokenizer.save_pretrained("./clip-vit-base-patch32")