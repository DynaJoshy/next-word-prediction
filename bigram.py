
from transformers import pipeline

pipe = pipeline("text-generation", model="your-model")

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("your-model")
model = AutoModelForCausalLM.from_pretrained("your-model")

text='textssssnvjnjfninif'


inputs = tokenizer(text, return_tensors="pt")


with torch.no_grad():  
    outputs = model.generate(
        inputs["input_ids"],  
        max_new_tokens=50,  
        num_return_sequences=1  
    )


generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)


print(generated_text)
