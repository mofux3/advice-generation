import torch
import transformers
import pandas as pd
import pyreft
from colorama import init, Fore

# Initialize Colorama for colored output
# 色付きの出力のためにColoramaを初期化
init()

# プリトレーニング済みモデルを読み込む関数
def load_pretrained_model():
    # Function to load pretrained model
    # プリトレーニング済みモデルを読み込む関数
    model_name = 'meta-llama/Llama-2-7b-chat-hf'
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map='cuda' if torch.cuda.is_available() else 'cpu',
        cache_dir='./workspace',
        token='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'
    )
    return model

# トークナイザを読み込む関数
def load_tokenizer(model_name):
    # Function to load tokenizer
    # トークナイザを読み込む関数
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name,
        model_max_tokens=2048,
        use_fast=False,
        padding_side="right",
        token='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'
    )
    tokenizer.pad_token = tokenizer.unk_token
    return tokenizer

# プロンプトテンプレートを定義する関数
def prompt_template(prompt):
    # Define Prompt Template
    # プロンプトのテンプレートを定義する
    return f"""<s>[INST]<<sys>>You are an assistant with high knowledge in civil engineering. You will create reports 
    based on pavements cracks and other defects detected on the street. You will create a report that includes the following:
    The kind of defect in question, what the consequences will be in 1 year if nothing is done, and the recommended 
    actions to be taken to solve the defect. Based the report on the following context provided: <</sys>>
        {prompt}
        [/INST]"""

# モデルを読み込み、トレーニングする関数
def load_and_train_model(model, tokenizer):
    # Load Data
    # データを読み込む
    df = pd.read_csv('your_dataset')
    X = df['target'].values
    y = df['goal'].values

    # Reft Configuration
    # Reftの設定
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
    # Reftモデルの初期化
    reft_model = pyreft.get_reft_model(model, reft_config)
    reft_model.set_device(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Prepare Data Module
    # データモジュールを準備する
    data_module = pyreft.make_last_position_supervised_data_module(
        tokenizer,
        model,
        [prompt_template(x) for x in X],
        y
    )

    # Training Arguments
    # トレーニングの引数
    training_arguments = transformers.TrainingArguments(
        num_train_epochs=10,
        output_dir='./models',
        per_device_train_batch_size=2,
        learning_rate=2e-3,
        logging_steps=20
    )

    # Reft Trainer
    # Reftトレーナー
    trainer = pyreft.ReftTrainerForCausalLM(
        model=reft_model,
        tokenizer=tokenizer,
        args=training_arguments,
        **data_module
    )

    # Train the Model
    # モデルをトレーニングする
    trainer.train()

    # Save the Model
    # モデルを保存する
    reft_model.set_device('cpu')
    reft_model.save(save_directory='./rude')

# Main Function
# メイン関数
if __name__ == "__main__":
    print("Loading pretrained model...")
    # プリトレーニング済みモデルを読み込んでいます...
    model = load_pretrained_model()
    print("Pretrained model loaded.")
    # プリトレーニング済みモデルを読み込みました.

    print("Loading tokenizer...")
    # トークナイザを読み込んでいます...
    tokenizer = load_tokenizer('meta-llama/Llama-2-7b-chat-hf')
    print("Tokenizer loaded.")
    # トークナイザを読み込みました.

    print("Models and data loaded. Starting training...")
    # モデルとデータが読み込まれました。トレーニングを開始します...
    load_and_train_model(model, tokenizer)
    print("Training completed. Model saved.")
    # トレーニングが完了しました。モデルが保存されました.
