from trl import SFTTrainer
from transformers import TrainingArguments

def train_model(model, tokenizer, train_dataset, dataset_text_field, max_seq_length, dataset_num_proc, packing, training_args):
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field=dataset_text_field,
        max_seq_length=max_seq_length,
        dataset_num_proc=dataset_num_proc,
        packing=packing,
        args=TrainingArguments(**training_args),
    )
    
    # Train the model
    train_results = trainer.train()
    
    # Optionally, you can return more specific training information if necessary
    return train_results
