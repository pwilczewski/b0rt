import torch
from transformers import BertTokenizer, BertForMaskedLM, BertConfig
from torch.utils.data import DataLoader
from transformers import LineByLineTextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import pipeline

# man this transformers library is kind of sketchy!

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('b0rt')
model = BertForMaskedLM.from_pretrained('b0rt')

# freeze every layer except the embedding layer
for param in model.parameters():
    if param.shape==torch.Size([30522,768]):
        continue
    else:
        param.requires_grad = False

# Load the training data
# dictionary with input_ids
train_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,file_path='training_data.txt', block_size=128
)

# collator takes input_ids, generates attention masks and masks
# labels are know from tokenizer?
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

import torch
import torch.nn.functional as F
max_length = 128

# input_ids are token_ids
# labels are the output variables
def custom_collate_fn(samples):
    # input_ids, labels = zip(*samples) # Unpack the samples
    input_ids = torch.stack(samples)
    labels = torch.stack(samples)

    # Pad the input ids and labels to the same length
    input_ids = F.pad(input_ids, pad=(0, 0, 0, input_ids.size(1) - max(input_ids.size(1), labels.size(1))), value=0)
    labels = F.pad(labels, pad=(0, 0, 0, labels.size(1) - max(input_ids.size(1), labels.size(1))), value=-100)

    # Create an attention mask
    attention_mask = input_ids != 0

    # Create a mask tensor
    mask = F.one_hot(input_ids, num_classes=tokenizer.vocab_size)

    # Randomly mask some tokens with a given probability
    mask_prob = 0.15
    mask = F.dropout(mask, p=mask_prob, training=True)

    # Return the batch of data
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask, "mask": mask}


# is my code even using dataloader?
# train_dataloader = DataLoader(
    #train_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate_fn)

training_args = TrainingArguments(output_dir='./bert_finetuned_logs', num_train_epochs=1)

# Initialize the Trainer
trainer = Trainer(model=model,args=training_args,
    data_collator=custom_collate_fn,train_dataset=train_dataset,)
# data_collator=data_collator

# Start training
trainer.train()

# Save the trained model
trainer.save_model('./bert_finetuned1')

# can I use a pytorch trainer?


plz = pipeline('fill-mask', model='bert_finetuned1')
