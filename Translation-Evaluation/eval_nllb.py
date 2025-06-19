import sys
import argparse
from datasets import load_dataset
from compute_metrics import compute_metrics,compute_comet_score
from utils import (
    display_evaluation,
    extract_sentences,
    generate_nllb_translations,
    google_translate_sentences
)
from comet import download_model, load_from_checkpoint
from dotenv import load_dotenv

# Load environment variables (for GOOGLE_API_KEY)
load_dotenv()

METADATA = {
    'nso': ['devtest'],
    'zul': ['devtest'],
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--lang_code', choices=METADATA.keys(), required=True)
    parser.add_argument('-s', '--split', required=True)
    return parser.parse_args()

def main():
    args = get_args()
    lang_code = args.lang_code
    split = args.split

    if split not in METADATA[lang_code]:
        sys.exit(f'{lang_code} does not support the split "{split}".')

    ds_src = load_dataset("openlanguagedata/flores_plus", "eng_Latn", split="devtest")
    ds_flores101_test = load_dataset("gsarti/flores_101", f"{lang_code}", split="devtest")
    ds_floresplus_test = load_dataset("openlanguagedata/flores_plus", f"{lang_code}_Latn", split="devtest")  
    src_text=extract_sentences(ds_src)
    ref_flores101_text=extract_sentences(ds_flores101_test)
    ref_floresplus_text=extract_sentences(ds_floresplus_test)

    tgt_lang_code = f"{lang_code}_Latn"
    
    pred_nllb = generate_nllb_translations(src_text, tgt_lang_code)
    pred_gt = google_translate_sentences(src_text,lang_code)

    comet_path = download_model("masakhane/africomet-stl-1.1")
    comet_model = load_from_checkpoint(comet_path)

    bleu_nllb_flores101, ter_nllb_flores101 = compute_metrics(ref_flores101_text, pred_nllb)
    bleu_nllb_floresplus, ter_nllb_floresplus = compute_metrics(ref_floresplus_text, pred_nllb)

    comet_nllb_flores101 = compute_comet_score(src_text, ref_flores101_text, pred_nllb, comet_model)
    comet_nllb_floresplus = compute_comet_score(src_text, ref_floresplus_text, pred_nllb, comet_model)

    bleu_gt_flores101, ter_gt_flores101 = compute_metrics(ref_flores101_text, pred_gt)
    bleu_gt_floresplus, ter_gt_floresplus = compute_metrics(ref_floresplus_text, pred_gt)

    comet_gt_flores101 = compute_comet_score(src_text, ref_flores101_text, pred_gt, comet_model)
    comet_gt_floresplus = compute_comet_score(src_text, ref_floresplus_text, pred_gt, comet_model)

    display_evaluation(
        lang_code=lang_code,
        split=split,
        scores_nllb=(bleu_nllb_flores101, bleu_nllb_floresplus, ter_nllb_flores101, ter_nllb_floresplus, comet_nllb_flores101, comet_nllb_floresplus),
        scores_google=(bleu_gt_flores101, bleu_gt_floresplus, ter_gt_flores101, ter_gt_floresplus, comet_gt_flores101, comet_gt_floresplus),
        src=src_text,
        ref=ref_floresplus_text,
        nllb_pred=pred_nllb,
        google_pred=pred_gt
    )

if __name__ == '__main__':
    main()
