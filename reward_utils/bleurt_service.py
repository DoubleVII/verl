import json
import euler
import argparse
from typing import List
from euler import base_compat_middleware
import warnings
import os
import tempfile
from typing import Dict

euler.install_thrift_import_hook()
from utils.idl.bleurt_thrift import BleurtService, BleurtReq



class BleurtClient:
    def __init__(self,
                 model='bleurt20',
                 cluster='lq'):
        target = f'sd://lab.mt.bleurt?cluster={model}&idc={cluster}'
        self.rpc_client = euler.Client(BleurtService,
                                       target=target,
                                       timeout=5000)
        self.rpc_client.use(base_compat_middleware.client_middleware)

    def score(self,
              hypo_texts: List[str],
              ref_texts: List[str]):
        assert len(hypo_texts) == len(ref_texts)
        req = BleurtReq(reference_list=ref_texts,
                        candidate_list=hypo_texts)
        resp = self.rpc_client.score(req)

        return resp.score_list


def preprocess_text(text: str):
    if "\n" in text:
        warnings.warn("output multiple lines, concatenate to one line.")
        text = text.replace("\n", " ")
    return text


def _to_model(bleurt_path: str) -> str:
    # Map bleurt_path to service model name; default to 'bleurt20'
    if not bleurt_path:
        return 'bleurt20'
    path_lower = str(bleurt_path).lower()
    if 'bleurt20' in path_lower or 'bleurt-20' in path_lower or '20' in path_lower:
        return 'bleurt20'
    return 'bleurt20'


def func_call(bleurt_path, mt_list, ref_list):
    assert len(mt_list) == len(ref_list)
    mt_proc = [preprocess_text(s) for s in mt_list]
    ref_proc = [preprocess_text(s) for s in ref_list]

    client = BleurtClient(model=_to_model(bleurt_path))
    scores = client.score(hypo_texts=mt_proc, ref_texts=ref_proc)
    if len(scores) != len(mt_list):
        raise ValueError(
            f"BLEURT output scores count ({len(scores)}) does not match input count ({len(mt_list)})"
        )
    return {"scores": scores}


def main(test_file: str, bleurt_path: str = "BLEURT-20") -> Dict:
    mt_list = []
    ref_list = []
    with open(test_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except Exception:
                warnings.warn(f"Invalid JSON line: {line}")
                continue
            mt = data.get("candidate")
            ref = data.get("reference")
            if mt is None or ref is None:
                warnings.warn(f"Missing candidate/reference fields: {line}")
                continue
            mt_list.append(mt)
            ref_list.append(ref)

    return func_call(bleurt_path, mt_list, ref_list)


if __name__ == '__main__':
    hypo_texts = ['你好啊世界！']
    ref_texts = ['你好世界！']

    bleurt_client = BleurtClient()
    bleurt_scores = bleurt_client.score(hypo_texts=hypo_texts,
                                        ref_texts=ref_texts)
    for hypo_text, ref_text, bleurt_score in zip(hypo_texts, ref_texts, bleurt_scores):
        json_data = dict(hypo_text=hypo_text,
                         ref_text=ref_text,
                         bleurt_score=bleurt_score)
        print(json.dumps(json_data, ensure_ascii=False))
