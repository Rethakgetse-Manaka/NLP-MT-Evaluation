import sacrebleu
import torch
torch.set_float32_matmul_precision('high')

def compute_metrics(ref, pred):
    bleu = sacrebleu.corpus_bleu(pred, [ref]).score
    ter = sacrebleu.corpus_ter(pred, [ref]).score
    return bleu, ter

def compute_comet_score(src, ref, pred, model):
    data = [{"src": s, "ref": r, "mt": p} for s, r, p in zip(src, ref, pred)]
    result = model.predict(data, batch_size=16, gpus=1 if torch.cuda.is_available() else 0)
    return float(result.system_score)