#!/usr/bin/env python3
"""Exercise the public CLI with a complete, small on-disk HF checkpoint."""

from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile

PROJECT_TMP = Path(__file__).resolve().parents[2] / "tmp"
PROJECT_TMP.mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = str(PROJECT_TMP)
tempfile.tempdir = str(PROJECT_TMP)

import torch
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace


def main():
    root = Path(__file__).resolve().parents[1]
    work = Path(tempfile.mkdtemp(prefix="plena-hf-graph-cli."))
    modeldir = work / "model"
    torch.manual_seed(456)
    config = LlamaConfig(
        vocab_size=96,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        attn_implementation="eager",
    )
    LlamaForCausalLM(config).half().eval().save_pretrained(modeldir)
    vocab = {"[UNK]": 0, "The": 1, "capital": 2, "of": 3, "France": 4, "is": 5}
    vocab.update({f"token{i}": i for i in range(6, 96)})
    tok = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]").save_pretrained(
        modeldir
    )
    subprocess.run(
        [
            sys.executable,
            str(root / "tools/plena-compile-model/plena_compile_model.py"),
            "--hf-model",
            str(modeldir),
            "--output-dir",
            str(work / "bundle"),
            "--prompt",
            "The capital of France is",
            "--execute",
            "--lp6-size",
            "64MiB",
        ],
        check=True,
    )
    report = json.loads((work / "bundle/compilation.json").read_text())
    assert (
        report["full_model"]
        and report["operation_driven"]
        and not report["model_specialized_backend"]
    )
    comparison = json.loads((work / "bundle/hf_comparison.json").read_text())
    assert comparison["allclose"]
    print("public CLI verified:", work, comparison)


if __name__ == "__main__":
    main()
