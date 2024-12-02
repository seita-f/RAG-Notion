import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rag_utils.utils import load_config


# YAMLファイルを読み込む
config = load_config("config.yml")
local_dir = config["LLM_MODEL"]["PHI"]["LOCAL_DIR"]
model_path = config["LLM_MODEL"]["PHI"]["MODEL_PATH"]


# モデルとトークナイザーをロード
# model = AutoModelForCausalLM.from_pretrained(
#     "microsoft/phi-1_5", 
#     trust_remote_code=True, 
#     torch_dtype=torch.float32,
#     cache_dir=local_dir
# )

# tokenizer = AutoTokenizer.from_pretrained(
#     "microsoft/phi-1_5", 
#     trust_remote_code=True, 
#     cache_dir=local_dir
# )

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.float32
)

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)

print("loaded model and tokenizer")

inputs = tokenizer(
    '''
    What is a capital city of Poland?
    ''', 
    return_tensors="pt"
)

# CPUでは.to(device)を省略（CPUで動作）
print("inputs is excecuted")

# テキスト生成
outputs = model.generate(**inputs, max_length=200)

print("outputs is executed")

# 結果をデコードして表示
text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
print(text)
