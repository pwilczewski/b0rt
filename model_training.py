import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForMaskedLM
from transformers import LineByLineTextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert_orig')
model = BertForMaskedLM.from_pretrained('bert_orig')

# freeze every layer except the embedding layer
for param in model.parameters():
    if param.shape==torch.Size([30522,768]):
        continue
    else:
        param.requires_grad = False

train_dataset = LineByLineTextDataset(tokenizer=tokenizer,file_path='training_data.txt', block_size=128)

# set up collator for MLM
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# train for 3 epochs
training_args = TrainingArguments(output_dir='./bert_finetuned_logs', 
                                  num_train_epochs=3, per_device_train_batch_size=64)

trainer = Trainer(model=model,args=training_args,data_collator=data_collator,train_dataset=train_dataset,)

# trainer.train()
# trainer.save_model('./bert_finetuned')
