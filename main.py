from configs.config import model, defaults, gpt_api, gemini_api
import sys


method = sys.argv[1]
dataset =  sys.argv[2]
output = sys.argv[3]


if model == "llava":
    from methods.LLaVA.llava_predictor import generate
    generate(defaults[dataset],output)
if model == "gemini":
    from methods.gemini.gemini_predictor import generate
    generate(defaults[dataset],output,gemini_api)
if model == "chatgpt":
    from methods.chatgpt.gpt_predictor  import generate
    generate(defaults[dataset],output,gpt_api)

