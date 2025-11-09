from transformers import AutoTokenizer, AutoModel

model_name = "CIDAS/clipseg-rd64-refined"

# Downloads model + tokenizer into the Hugging Face cache (~/.cache/huggingface)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Save them into a directory
save_dir = "./pretrain/clipseg-rd64-refined-local"
tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)