import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import load_config


# YAMLファイルを読み込む
config = load_config("config.yml")


local_dir = config["LLM_MODEL"]["PHI"]["LOCAL_DIR"]

# デバイスの選択
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデルとトークナイザーをロード
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-1_5", 
    trust_remote_code=True, 
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    cache_dir=local_dir
).to(device)

tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/phi-1_5", 
    trust_remote_code=True, 
    cache_dir=local_dir
)

# 入力データをトークナイズしてデバイスに移動
inputs = tokenizer(
    '''```python

def print_prime(n):

   """

   Print all primes between 1 and n

   """''', 
    return_tensors="pt"
).to(device)

# テキスト生成
outputs = model.generate(**inputs, max_length=200)

# 結果をデコードして表示
text = tokenizer.batch_decode(outputs)[0]
print(text)
