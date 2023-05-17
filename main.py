import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import csv

class ChatDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file):
        self.conversations = []
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.max_len = self.tokenizer.model_max_length

        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                self.conversations.append(row)

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        conversation = self.conversations[idx]
        input_ids = []
        label_ids = []

        for i in range(0, len(conversation) - 1, 2):
            user, message = conversation[i], conversation[i + 1]
            user_tokens = self.tokenizer.encode(message, add_special_tokens=False, truncation=True,
                                                max_length=self.max_len - 1)
            input_ids.extend([self.tokenizer.bos_token_id] + user_tokens)
            label_ids.extend(user_tokens + [self.tokenizer.eos_token_id])

        input_ids = input_ids[:self.max_len]
        label_ids = label_ids[:self.max_len]

        input_ids = torch.nn.functional.pad(torch.tensor(input_ids), pad=(0, self.max_len - len(input_ids)))
        label_ids = torch.nn.functional.pad(torch.tensor(label_ids), pad=(0, self.max_len - len(label_ids)))

        return input_ids, label_ids


# Load the chat dataset from CSV
csv_file = 'personality.csv'

# Create the chat dataset
chat_dataset = ChatDataset(csv_file)

# Set up tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Prepare the chat dataset
dataset = ChatDataset(csv_file)

# Set the batch size and create the data loader
batch_size = 2
chat_dataloader = DataLoader(chat_dataset, batch_size=batch_size, shuffle=True)

# Set the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the GPT-2 model
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.to(device)

# Training loop
epochs = 5
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

for epoch in range(epochs):
    epoch_loss = 0
    for batch in chat_dataloader:
        input_ids, label_ids = batch
        input_ids = input_ids.to(device)
        label_ids = label_ids.to(device)

        outputs = model(input_ids, labels=label_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss += loss.item()

    print(f"Epoch: {epoch+1} Loss: {epoch_loss}")

# Save the trained model
model.save_pretrained("C:\\Users\\bsvar\\OneDrive\\Documents\\tokenpretrained")
tokenizer.save_pretrained("C:\\Users\\bsvar\\OneDrive\\Documents\\chatbotmodelpretrained")
