from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name = "google/gemma-2-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

generation_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=4000, device=0)
llm = HuggingFacePipeline(pipeline=generation_pipeline)

