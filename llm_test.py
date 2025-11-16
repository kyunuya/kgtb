import os
import re
import json
import time
import argparse
from datetime import datetime, timedelta
from typing import List, Dict

import torch
import pandas as pd
from tqdm import tqdm
from unsloth import FastLanguageModel
from peft import PeftModel
from transformers import AutoTokenizer

BASE_MODEL_NAME = "unsloth/Qwen3-4B-unsloth-bnb-4bit"
MAX_SEQ_LENGTH = 2048


def format_prompt_for_inference(user_text: str) -> str:
    return (
        "<|im_start|>system\n"
        "You are a helpful assistant for next POI recommendation.<|im_end|>\n"
        "<|im_start|>user\n" + user_text + "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def extract_prediction(text: str) -> str:
    text = " ".join(text.strip().split())  # normalize whitespace
    m = re.search(r"\b(POI|Category|Region)\s+stru_\d+_\d+_\d+\b", text)
    if m:
        return m.group(0)
    # Fallback to ID-only
    m2 = re.search(r"\bstru_\d+_\d+_\d+\b", text)
    if m2:
        return m2.group(0)
    return text.strip()


def normalize_answer(s: str) -> str:
    return " ".join(s.strip().split())


def load_finetuned_model_and_tokenizer(adapter_dir: str, base_model_name: str):
    print(f"Loading base model: {base_model_name}")
    base_model, _ = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
        trust_remote_code=True,
    )
    print(f"Loading LoRA adapters from: {adapter_dir}")
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    FastLanguageModel.for_inference(model)

    print("Loading tokenizer from the fine-tuned directory for consistency…")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Qwen3 against filtered_{LOCATION}_test.csv")
    parser.add_argument("--location", required=True, choices=["CA", "NYC", "TKY"], help="Dataset tag used in filenames")
    parser.add_argument("--model_dir", default=None, help="Path to fine-tuned adapter/tokenizer dir; default ./kgtb_llm_{LOCATION}_finetuned")
    parser.add_argument("--output_dir", default="./kgtb_results", help="Directory to write evaluation results")
    parser.add_argument("--max_eval_samples", type=int, default=500, help="Max number of (user,t) samples to evaluate (cap)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")

    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)

    args = parser.parse_args()

    location = args.location
    model_dir = args.model_dir or f"./kgtb_llm_{location}_finetuned"
    os.makedirs(os.path.join(args.output_dir, location), exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.output_dir, location, f"eval_{timestamp}.json")

    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Fine-tuned adapter/tokenizer not found at {model_dir}. Run train_llm.py first.")

    print(f"Loading model from adapter: {model_dir}")
    model, tokenizer = load_finetuned_model_and_tokenizer(model_dir, BASE_MODEL_NAME)

    stru_map_path = f"stru_ids_{location}.json"
    if not os.path.exists(stru_map_path):
        raise FileNotFoundError(f"Stru IDs mapping not found at {stru_map_path}.")
    with open(stru_map_path, "r") as f:
        raw_to_token = json.load(f)

    test_csv_path = f"kgtb/data/dataset/{location}/filtered_{location}_test.csv"
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Test CSV not found at {test_csv_path}.")
    test_df = pd.read_csv(test_csv_path)
    if "Local Time" not in test_df.columns:
        raise ValueError("Expected 'Local Time' column in test CSV.")
    test_df["Local Time dt"] = pd.to_datetime(test_df["Local Time"])

    def to_token(raw):
        if pd.isna(raw):
            return None
        return raw_to_token.get(str(raw))

    samples: List[Dict] = []
    for uid, g in test_df.groupby("Uid"):
        g = g.sort_values("Local Time dt").reset_index(drop=True)
        for i in range(1, len(g)):
            hist = g.iloc[:i]
            target = g.iloc[i]

            user_tok = to_token(uid)
            if not user_tok:
                continue

            seen = {}
            for _, row in hist.iterrows():
                seen[str(row["Pid"])] = (row["Catname"], row["Region"])  # last occurrence
            pref_parts = []
            for pid_str, (cat, reg) in seen.items():
                poi_tok = to_token(pid_str)
                cat_tok = to_token(cat)
                reg_tok = to_token(reg)
                if poi_tok and cat_tok and reg_tok:
                    pref_parts.append(f"{cat_tok} {reg_tok} {poi_tok}")
            if not pref_parts:
                continue
            pref_pois = ", ".join(pref_parts)

            traj_parts = []
            for _, row in hist.iterrows():
                poi_tok = to_token(row["Pid"])
                cat_tok = to_token(row["Catname"])
                reg_tok = to_token(row["Region"])
                t = row["Local Time"]
                if poi_tok and cat_tok and reg_tok:
                    traj_parts.append(f"visiting {cat_tok} {reg_tok} {poi_tok} at time {t}")
            if not traj_parts:
                continue
            traj = ", ".join(traj_parts)

            t_idx = len(hist) + 1
            input_text = (
                f"Please conduct a next POI recommendation. There is user {user_tok} "
                f"and his preferable POIs: {pref_pois}. Here is his current trajectory: {traj}. "
                f"Which POI will the user {user_tok} visit at time t{t_idx}?"
            )

            target_poi_tok = to_token(target["Pid"])
            if not target_poi_tok:
                continue

            samples.append({
                "input": input_text,
                "target_stru": target_poi_tok,
            })

            if args.max_eval_samples and len(samples) >= args.max_eval_samples:
                break
        if args.max_eval_samples and len(samples) >= args.max_eval_samples:
            break

    if not samples:
        raise RuntimeError("No evaluable samples could be constructed from test CSV and stru mapping.")

    print(f"Evaluating {len(samples)} samples…")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results: List[Dict] = []
    correct = 0
    start_t = time.time()

    for i, ex in enumerate(tqdm(samples, desc="Eval")):
        user = ex["input"]
        target_stru = ex["target_stru"]

        prompt = format_prompt_for_inference(user)
        enc = tokenizer([prompt], return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH - args.max_new_tokens)
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

        gen_tokens = output_ids[0][enc["input_ids"].shape[1]:]
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

        pred_compact = extract_prediction(gen_text)
        m_pred = re.search(r"stru_\d+_\d+_\d+", pred_compact)
        pred_stru = m_pred.group(0) if m_pred else pred_compact.strip()
        is_correct = (pred_stru == target_stru)
        correct += int(is_correct)

        results.append({
            "idx": i,
            "input": user,
            "target_stru": target_stru,
            "generated": gen_text.strip(),
            "prediction": pred_compact,
            "pred_stru": pred_stru,
            "correct": bool(is_correct),
        })

    elapsed = time.time() - start_t
    n = len(samples)
    metrics = {
        "samples": n,
        "stru_accuracy": correct / n if n else 0.0,
        "seconds": elapsed,
    }

    output = {
        "info": {
            "adapter_dir": model_dir,
            "base_model": BASE_MODEL_NAME,
            "location": location,
            "test_csv": test_csv_path,
            "stru_map": stru_map_path,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
        "metrics": metrics,
        "results": results,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print("\n=== Evaluation Summary ===")
    print(f"Saved: {out_path}")
    print(f"Samples: {metrics['samples']}")
    print(f"Stru token accuracy: {metrics['stru_accuracy']:.4f}")


if __name__ == "__main__":
    main()
