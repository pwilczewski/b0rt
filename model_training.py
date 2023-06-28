import torch
from transformers import BertTokenizer, BertForMaskedLM, BertConfig
from torch.utils.data import DataLoader
from transformers import LineByLineTextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import pipeline

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('b0rt')
model = BertForMaskedLM.from_pretrained('b0rt')

# add a layer 768x768 to train, skip connection!
# try LoRA

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
