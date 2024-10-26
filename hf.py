from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

# Authenticate using the token directly in the code
login("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")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_auth_token=True)
