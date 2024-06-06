# Nakahara-Labs 向けアドバイス生成用 Github リポジトリ（正確な名前はわかりませんが、後で変更します）

このリポジトリには、Huggingface と Stanford のライブラリを使用してモデルをクエリおよびファインチューニングするためのコードスニペットなどが含まれます。また、RAG を使用してモデルを推論するためのものも含まれます。

## 必要条件
- 有効なアクセス トークンを持つ Huggingface アカウント。
- OpenAI の API トークン（モデルの推論などに OpenAI の API を使用する場合）。

## セットアップ
1. リポジトリをクローンします
    ```bash
    git clone https://github.com/shiorix3/advice-generation
    cd advice-generation
    ```

2. 依存関係をインストールします
    ```bash
    pip install -r requirements.txt
    ```

3. アクセストークンを設定します
   必要なアクセストークンが設定されていることを確認してください。環境変数に設定するか、スクリプトに直接追加することができます。

   例:
   ```bash
   huggingface-cli login
   ```

4. スクリプトを実行します
   クエリ、ファインチューニング、または推論などの特定のタスクについては、リポジトリ内の個々のスクリプトを参照してください。

## 貢献ガイドライン
1. リポジトリをフォークします。
2. 新しいブランチを作成します (`git checkout -b feature-branch`)。
3. 変更を行います。
4. 変更をコミットします (`git commit -m 'Add some feature'`)。
5. ブランチをプッシュします (`git push origin feature-branch`)。
6. プルリクエストを作成します。

## お問い合わせ先
ご質問やサポートが必要な場合は、問題ページで問題を作成してください。


# Github Repo for Advice Generation for Nakahara-Labs (Not sure if this is the correct name, will change later)

This repo will contain code snippets and the like for querying and fine-tuning models using Huggingface's and Stanford's libraries, as well as inferencing models using RAG.

## Requirements
- Huggingface account with a valid access token.
- OpenAI API token (if you want to use OpenAI's API for inferencing models or other usages).

## Setup
1. Clone the repository
    ```bash
    git clone https://github.com/shiorix3/advice-generation
    cd advice-generation
    ```

2. Install dependencies
    ```bash
    pip install -r requirements.txt
    ```

3. Configure access tokens
   Ensure you have the necessary access tokens configured. You can do this by setting environment variables or adding them directly to the scripts.

   Example:
   ```bash
   huggingface-cli login
   ```

4. Run scripts
   Refer to individual scripts in the repo for specific tasks like querying, fine-tuning, or inferencing.

## Contribution Guidelines
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a pull request.

## Contact
For any queries or support, please create an issue on the issues page.
