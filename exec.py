import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the pretrained tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("C:\\Users\\bsvar\\OneDrive\\Documents\\tokenpretrained")
model = GPT2LMHeadModel.from_pretrained("C:\\Users\\bsvar\\OneDrive\\Documents\\chatbot1")

# Set the model to evaluation mode
model.eval()

# Conversation loop
while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break

    # Encode the user input and generate the attention mask
    input_ids = tokenizer.encode(user_input, add_special_tokens=True, return_tensors="pt")
    attention_mask = torch.ones_like(input_ids)

    # Generate the model's response with increased randomness
    with torch.no_grad():
        output = model.generate(input_ids, attention_mask=attention_mask, max_length=100, temperature=0.8)

    # Decode and print the response
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print("SnipChat: " + response)
