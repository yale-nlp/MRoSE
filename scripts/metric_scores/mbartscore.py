import torch
import torch.nn as nn
import traceback
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from typing import List, Optional, Dict
import numpy as np
from base_metric import Metric
import json


class MBARTScorer:
    def __init__(self, device: str, max_length=1024, checkpoint="facebook/mbart-large-50"):
        # Set up model
        self.device = device
        self.max_length = max_length
        self.tokenizer = MBart50TokenizerFast.from_pretrained(checkpoint)
        self.model = MBartForConditionalGeneration.from_pretrained(checkpoint)
        self.model.eval()
        self.model.to(device)

        # Set up loss
        self.loss_fct = nn.NLLLoss(reduction='none', ignore_index=self.model.config.pad_token_id)
        self.lsm = nn.LogSoftmax(dim=1)

    def score(self, srcs, tgts, batch_size=4):
        """ Score a batch of examples """
        score_list = []
        for i in range(0, len(srcs), batch_size):
            src_list = srcs[i: i + batch_size]
            tgt_list = tgts[i: i + batch_size]
            try:
                with torch.no_grad():
                    encoded_src = self.tokenizer(
                        src_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    encoded_tgt = self.tokenizer(
                        tgt_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    src_tokens = encoded_src['input_ids'].to(self.device)
                    src_mask = encoded_src['attention_mask'].to(self.device)

                    tgt_tokens = encoded_tgt['input_ids'].to(self.device)
                    tgt_mask = encoded_tgt['attention_mask']
                    tgt_len = tgt_mask.sum(dim=1).to(self.device)

                    output = self.model(
                        input_ids=src_tokens,
                        attention_mask=src_mask,
                        labels=tgt_tokens
                    )
                    logits = output.logits.view(-1, self.model.config.vocab_size)
                    loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
                    loss = loss.view(tgt_tokens.shape[0], -1)
                    loss = loss.sum(dim=1) / tgt_len
                    curr_score_list = [-x.item() for x in loss]
                    score_list += curr_score_list

            except RuntimeError:
                traceback.print_exc()
                print(f'source: {src_list}')
                print(f'target: {tgt_list}')
                exit(0)
        return score_list



class MBARTScore(Metric):
    def __init__(self, device=0, max_length=256, checkpoint='facebook/mbart-large-cc25'):
        self.scorer = MBARTScorer(device=f"cuda:{device}", max_length=max_length, checkpoint=checkpoint)

    def evaluate_batch(self, summaries: List[str], batch_size: int, articles: Optional[List[str]], references: Optional[List[str]]) -> List[Dict]:
        scores_recall = self.scorer.score(summaries, references, batch_size=batch_size)
        scores_precision = self.scorer.score(references, summaries, batch_size=batch_size)
        return [{
            'bartscore_f1': 2 * score_r * score_p / (score_r + score_p + 1e-10),
            'bartscore_r': score_r,
            'bartscore_p': score_p
        } for score_r, score_p in zip(scores_recall, scores_precision)]




    