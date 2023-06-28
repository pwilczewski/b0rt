import torch
from transformers import BertTokenizer, BertForMaskedLM, BertConfig
from torch.utils.data import DataLoader
from transformers import LineByLineTextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import pipeline

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('b0rt')
model = BertForMaskedLM.from_pretrained('b0rt')

# try LoRA


# add a layer 768x768 to train, skip connection!
import torch.nn as nn
embeddings = model.get_input_embeddings()

# Method 2: using custom class
class b0rt_ft(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # turn off graident
        x = self.model.get_input_embeddings()(x)
        x = nn.Linear(768, 768)
        x = nn.ReLU(x)

        x = self.model.classifier(x)
        return x

new_model = b0rt_ft()


# Method 1: using nn.Sequential
# model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg19', pretrained=True)
features = model.features
avgpool = model.avgpool
classifier = model.classifier
new_model = nn.Sequential(
    features,
    avgpool,
    nn.LayerNorm(avgpool.output_size),
    nn.Flatten(),
    classifier
)

# freeze every layer except the embedding layer
for param in model.parameters():
    if param.shape==torch.Size([30522,768]):
        continue
    else:
        param.requires_grad = False

train_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,file_path='training_data.txt', block_size=128)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

import torch
import torch.nn.functional as F

# is my code even using dataloader?
# train_dataloader = DataLoader(
    #train_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate_fn)

training_args = TrainingArguments(output_dir='./bert_finetuned_logs', 
                                  num_train_epochs=3, 
                                  per_device_train_batch_size=64)

# Initialize the Trainer
trainer = Trainer(model=model,args=training_args,
    data_collator=data_collator,train_dataset=train_dataset,)

# Start training
trainer.train()

# Save the trained model
trainer.save_model('./bert_finetuned1')

# can I use a pytorch trainer?
plz = pipeline('fill-mask', model='bert_finetuned1')
