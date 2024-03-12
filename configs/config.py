from pathlib import Path

gpt_api = ""
gemini_api = "" 

model = "llava"

current_path = Path.cwd()
parent_path = current_path.parent
datasets_path = parent_path / "datasets"

defaults = {
    "vqav2": str(datasets_path) + "/vqav2/vqav2_qa_formatted.json",
    "agvqa":str(datasets_path) + "/agvqa/agvqa_qa_formatted.json",
    "gqa":str(datasets_path) + "/gqa/gqa_qa_formatted.json",
    "okvqa":str(datasets_path) + "/okvqa/okvqa_qa_formatted.json",

}