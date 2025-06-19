# NLP-MT-Evaluation Workspace

This repository contains two main projects for evaluating machine translation (MT) systems and analyzing translation errors using large language models (LLMs):

- **Translation-Evaluation**: Scripts for evaluating translation quality using automatic metrics and reference datasets.
- **LLM Error Detection**: Jupyter notebook for classifying and explaining translation errors using OpenAI's GPT models.

---

## Table of Contents
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Getting API Keys](#getting-api-keys)
  - [HuggingFace Access Token](#huggingface-access-token)
  - [OpenAI API Key](#openai-api-key)
  - [Google Cloud Translation API Key](#google-cloud-translation-api-key)
- [Translation-Evaluation Usage](#translation-evaluation-usage)
- [LLM Error Detection Usage](#llm-error-detection-usage)
- [Evaluation Metrics Explained](#evaluation-metrics-explained)
- [Troubleshooting](#troubleshooting)

---

## Project Structure

```
.
├── Translation-Evaluation/
│   ├── compute_metrics.py
│   ├── eval_nllb.py
│   ├── requirements.txt
│   └── utils.py
├── LLM Error Detection/
│   └── llm-error-detection.ipynb
└── README.md
```

---

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd NLP-MT-Evaluation
   ```

2. **Set up a Python environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies for Translation-Evaluation:**
   ```bash
   pip install -r Translation-Evaluation/requirements.txt
   ```

4. **Install additional dependencies for LLM Error Detection (if not already installed):**
   ```bash
   pip install transformers accelerate openai huggingface_hub fsspec datasets
   ```

---

## Getting API Keys

### HuggingFace Access Token
Some scripts and notebooks require access to HuggingFace-hosted models and datasets.

1. Go to https://huggingface.co/join and create an account (if you don't have one).
2. After logging in, go to https://huggingface.co/settings/tokens
3. Click **New token**, give it a name, and select the `read` role.
4. Copy the generated token.
5. You can log in via the CLI or set the token as an environment variable:
   ```bash
   huggingface-cli login
   # Paste your token when prompted
   ```
   Or, in Python:
   ```python
   from huggingface_hub import login
   login('your_token_here')
   ```

### OpenAI API Key
To use OpenAI's GPT models for error detection:

1. Go to https://platform.openai.com/signup and create an account.
2. Navigate to https://platform.openai.com/api-keys
3. Click **Create new secret key** and copy the key.
4. In your notebook or script, use the key as follows:
   ```python
   import openai
   client = openai.OpenAI(api_key="<YOUR_API_KEY>")
   ```
   Or set it as an environment variable:
   ```bash
   export OPENAI_API_KEY="<YOUR_API_KEY>"
   ```

### Google Cloud Translation API Key
For Google Translate integration in `Translation-Evaluation`:

1. Go to https://console.cloud.google.com/
2. Create a project (or select an existing one).
3. Enable the **Cloud Translation API** for your project.
4. Go to **APIs & Services > Credentials** and click **Create credentials > API key**.
5. Copy the API key.
6. Save it in a `.env` file in the `Translation-Evaluation` folder:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   ```

---

## Translation-Evaluation Usage

This module evaluates translation quality using NLLB and Google Translate, comparing outputs against reference datasets with BLEU, TER, and COMET metrics.

### 1. Prepare Environment
- Ensure you have all required API keys and dependencies installed.

### 2. Run Evaluation
From the `Translation-Evaluation` directory, run:
```bash
python eval_nllb.py -l <lang_code> -s <split>
```
- `lang_code`: `nso` (Sepedi) or `zul` (Zulu)
- `split`: `devtest` (currently supported)

Example:
```bash
python eval_nllb.py -l zul -s devtest
```

### 3. Output
- The script will print evaluation scores (BLEU, TER, COMET) for both NLLB and Google Translate outputs, and show sample translations.

---

## LLM Error Detection Usage

This notebook uses OpenAI's GPT models to classify and explain translation errors between original and corrected translations.

### 1. Open the Notebook
- Launch Jupyter and open `LLM Error Detection/llm-error-detection.ipynb`.

### 2. Install Required Packages (if needed)
The first cells will prompt you to install any missing packages:
```python
!pip install transformers accelerate openai huggingface_hub fsspec datasets
```

### 3. Authenticate with HuggingFace
Run the cell:
```python
import huggingface_hub
huggingface_hub.login()  # Enter your token when prompted
```

### 4. Set Up OpenAI API Key
Edit the cell where `openai.OpenAI(api_key="<YOUR_API_KEY>")` is called, replacing `<YOUR_API_KEY>` with your actual key.

### 5. Run the Notebook
- The notebook will load datasets, align translations, and use GPT to classify and explain errors.
- Results are summarized and grouped by error type.

---

## Evaluation Metrics Explained
- **BLEU**: Measures n-gram overlap between machine translation and reference.
- **TER**: Translation Edit Rate; measures the number of edits needed to match the reference.
- **COMET**: Neural metric that predicts translation quality using source, reference, and hypothesis.

---

## Troubleshooting
- **CUDA/torch errors**: If you don't have a GPU, the scripts will fall back to CPU, but may run slower.
- **API authentication errors**: Double-check your API keys and ensure they are set in your environment or `.env` files.
- **Dataset download issues**: Ensure you are logged in to HuggingFace and have internet access.
- **Google Translate quota**: If you hit quota limits, check your Google Cloud billing and API usage.

---

## Contact
For questions or issues, please open an issue in this repository. 