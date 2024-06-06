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
