import torch
import transformers
import pandas as pd
import pyreft
from colorama import init, Fore

# Initialize Colorama for colored output
init()

# Function to load pretrained model
def load_pretrained_model():
    model_name = 'meta-llama/Llama-2-7b-chat-hf'
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map='cuda' if torch.cuda.is_available() else 'cpu',
        cache_dir='./workspace',
        token='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'
    )
    return model

# Function to load tokenizer
def load_tokenizer(model_name):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name,
        model_max_tokens=2048,
        use_fast=False,
        padding_side="right",
        token='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'
    )
    tokenizer.pad_token = tokenizer.unk_token
    return tokenizer

# Define Prompt Template
def prompt_template(prompt):
    return f"""<s>[INST]<<sys>>You are an assistant with high knowledge in civil engineering. You will create reports 
    based on pavements cracks and other defects detected on the street. You will create a report that includes the following:
    The kind of defect in question, what the consequences will be in 1 year if nothing is done, and the recommended 
    actions to be taken to solve the defect. Based the report on the following context provided: <</sys>>
        {prompt}
        [/INST]"""

# Function to load and train the model
def load_and_train_model(model, tokenizer):
    # Load Data
    df = pd.read_csv('data/supercool.csv')
    X = df['target'].values
    y = df['goal'].values

    # Reft Configuration
    reft_config = pyreft.ReftConfig(
        representations={
            "layer": 15,
            "component": "block_output",
            "low_rank_dimension": 4,
            "intervention": pyreft.LoreftIntervention(
                embed_dim=model.config.hidden_size, low_rank_dimension=4
            )
        }
    )

    # Initialize Reft Model
    reft_model = pyreft.get_reft_model(model, reft_config)
    reft_model.set_device(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Prepare Data Module
    data_module = pyreft.make_last_position_supervised_data_module(
        tokenizer,
        model,
        [prompt_template(x) for x in X],
        y
    )

    # Training Arguments
    training_arguments = transformers.TrainingArguments(
        num_train_epochs=10,
        output_dir='./models',
        per_device_train_batch_size=2,
        learning_rate=2e-3,
        logging_steps=20
    )

    # Reft Trainer
    trainer = pyreft.ReftTrainerForCausalLM(
        model=reft_model,
        tokenizer=tokenizer,
        args=training_arguments,
        **data_module
    )

    # Train the Model
    trainer.train()

    # Save the Model
    reft_model.set_device('cpu')
    reft_model.save(save_directory='./rude')

# Main Function
if __name__ == "__main__":
    print("Loading pretrained model...")
    model = load_pretrained_model()
    print("Pretrained model loaded.")

    print("Loading tokenizer...")
    tokenizer = load_tokenizer('meta-llama/Llama-2-7b-chat-hf')
    print("Tokenizer loaded.")

    print("Models and data loaded. Starting training...")
    load_and_train_model(model, tokenizer)
    print("Training completed. Model saved.")
