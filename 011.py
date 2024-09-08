from openai import OpenAI

API_BASE = "https://api.lingyiwanwu.com/v1"
API_KEY = "25e88c7998394b41b73d835110423542"
client = OpenAI(
  api_key=API_KEY,
  base_url=API_BASE
)
completion = client.chat.completions.create(
  model="yi-large",
  messages=[{"role": "user", "content": "Hi, who are you?"}],
  stream=True
)
for chunk in completion:
  print(chunk.choices[0].delta.content or "", end="", flush=True)