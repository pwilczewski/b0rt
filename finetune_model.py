import torch
from transformers import BertTokenizer, BertForMaskedLM
from transformers import LineByLineTextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert_orig')
model = BertForMaskedLM.from_pretrained('bert_orig')

# freeze layers not used for fine tuning
def freeze_parameters(model, layer_type, layer_no=11):
    if layer_type=="embedding_only":
        finetuning_layer_name = "bert.embeddings.word_embeddings.weight"
        finetuning_layer_shapes = [torch.Size([30522,768])]
    elif layer_type=="classification_only":
        finetuning_layer_name = "cls.predictions.transform.dense.weight"
        finetuning_layer_shapes = [torch.Size([768,768])]
    elif layer_type=="attention_only":
        finetuning_layer_name = "bert.encoder.layer." + str(layer_no) + "."
        finetuning_layer_shapes = [torch.Size([768,768]), torch.Size([3072, 768]), torch.Size([768, 3072])]

    for (name, param) in model.named_parameters():
        if (finetuning_layer_name in name) and (param.shape in finetuning_layer_shapes):
            continue
        else:
            param.requires_grad = False

    return model

model = freeze_parameters(model, "attention_only", layer_no=0)

train_dataset = LineByLineTextDataset(tokenizer=tokenizer,file_path='data/training_data.txt', block_size=128)

# set up collator for MLM
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# train for 3 epochs
training_args = TrainingArguments(output_dir='./bert_finetuned_logs', 
                                  num_train_epochs=3, per_device_train_batch_size=64)

trainer = Trainer(model=model,args=training_args,data_collator=data_collator,train_dataset=train_dataset,)

trainer.train()
trainer.save_model('./bert_attention_layer0_finetuned')
