import torch
torch.set_float32_matmul_precision('high')
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from google.cloud import translate_v2 as translate
from tabulate import tabulate

NLLB_MODEL_NAME = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL_NAME)
nllb_model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL_NAME).to("cuda")


def google_translate_sentences(sentences, target_lang_code):

    language_mapping = {
        "zul": "zu",  
        "nso": "nso", 
        
    }
    lang_code= language_mapping[target_lang_code]

    translate_client = translate.Client()
    translations = []

    for text in sentences:
        result = translate_client.translate(
            text,
            target_language=lang_code,
            source_language="en"
        )
        translations.append(result["translatedText"])
    print("=================================GNMT Translations complete=============================")
    return translations

def generate_nllb_translations(src_sentences, tgt_lang, batch_size=16):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = nllb_model.to(device)
    tokenizer.src_lang = "eng_Latn" 

    translations = []

    for i in range(0, len(src_sentences), batch_size):
        batch = src_sentences[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        
        bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)

        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=bos_token_id,
            max_length=512
        )
        batch_translations = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        translations.extend(batch_translations)
    print("=================================NLLB Translations complete=============================")
    return translations

def extract_sentences(dataset):
    if 'sentence' in dataset[0]:
        return [x['sentence'] for x in dataset]
    elif 'text' in dataset[0]:
        return [x['text'] for x in dataset]
    else:
        raise KeyError("Neither 'sentence' nor 'text' found in the dataset.")
    
def display_evaluation(
    lang_code,
    split,
    scores_nllb,
    scores_google,
    src,
    ref,
    nllb_pred,
    google_pred
):
    print("\n======= Combined Evaluation Results =======")
    print(f"Language: {lang_code} | Split: {split}\n")

    table = [
        ["BLEU", scores_nllb[0], scores_nllb[1], scores_google[0], scores_google[1]],
        ["TER", scores_nllb[2], scores_nllb[3], scores_google[2], scores_google[3]],
        ["COMET", scores_nllb[4], scores_nllb[5], scores_google[4], scores_google[5]],
    ]
    headers = [
        "Metric",
        "NLLB (Original)", "NLLB (Corrected)",
        "Google (Original)", "Google (Corrected)"
    ]
    print(tabulate(table, headers=headers, tablefmt="fancy_grid", floatfmt=".2f"))

    print("\n🔍 Sample Translations (up to 5):")
    for i in range(min(5, len(src))):
        print(f"\n🟦 English       : {src[i]}")
        print(f"🤖 NLLB Pred     : {nllb_pred[i]}")
        print(f"🌐 Google Pred   : {google_pred[i]}")
        print(f"🟩 Reference     : {ref[i]}")
    print("\n(Note: compared to corrected reference)")

