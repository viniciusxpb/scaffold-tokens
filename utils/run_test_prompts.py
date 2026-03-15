#!/usr/bin/env python3
"""
Runs the 53 test prompts and saves results to data/test_results/.
Then analyzes length control, structural planning, and statistics.

Usage:
    python utils/run_test_prompts.py [--checkpoint PATH]
"""

import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import argparse
import glob
import json
import sys
import torch
torch.set_num_threads(2)
import tiktoken

# Add project root to path to import inference
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference import load_checkpoint, generate, FF_BASE_ID

enc = tiktoken.get_encoding("gpt2")

RESULTS_DIR = "data/test_results"

# ── Prompts ──────────────────────────────────────────────────────────────

PROMPTS = [
    # Short (ff_50 to ff_99)
    (50, "O governo"),
    (50, "A economia"),
    (50, "Segundo especialistas"),
    (60, "O presidente"),
    (60, "A polícia"),
    (60, "De acordo com"),
    (70, "O mercado financeiro"),
    (70, "A cidade de São Paulo"),
    (80, "O ministro"),
    (80, "Nesta segunda"),
    (90, "A Folha apurou"),
    (90, "O ex-presidente"),
    (99, "De acordo com o governo"),
    (99, "A economia brasileira"),
    (99, "O deputado"),
    # Medium (ff_100 to ff_299)
    (100, "O governo federal"),
    (100, "A economia brasileira"),
    (100, "Segundo o ministro"),
    (150, "O presidente da República"),
    (150, "A polícia federal"),
    (150, "De acordo com a pesquisa"),
    (200, "O governo"),
    (200, "A economia"),
    (200, "Segundo especialistas"),
    (200, "O ex-presidente"),
    (200, "Nesta segunda"),
    (250, "O mercado financeiro"),
    (250, "De acordo com"),
    (250, "A Folha apurou"),
    (299, "O presidente"),
    (299, "A economia brasileira"),
    (299, "Segundo o ministro"),
    # Long (ff_300 to ff_630)
    (300, "O governo"),
    (300, "A economia brasileira"),
    (300, "O presidente da República"),
    (350, "Segundo especialistas"),
    (350, "De acordo com"),
    (350, "O ex-presidente"),
    (400, "O governo federal"),
    (400, "A economia"),
    (400, "Nesta segunda"),
    (450, "O mercado financeiro"),
    (450, "A Folha apurou"),
    (450, "O presidente"),
    (500, "O governo"),
    (500, "A economia brasileira"),
    (500, "De acordo com o governo"),
    (550, "O presidente da República"),
    (600, "A economia"),
    (630, "O governo"),
    # Stress
    (700, "O governo"),
    (800, "A economia"),
    (999, "O presidente"),
]


def count_generated_words(text: str, prompt: str) -> int:
    """Count generated words (excludes prompt)."""
    generated = text[len(prompt):].strip()
    if not generated:
        return 0
    return len(generated.split())


def extract_words(text: str, prompt: str) -> list[str]:
    """Extract all generated words."""
    generated = text[len(prompt):].strip()
    if not generated:
        return []
    return generated.split()


def run_prompts(model, device="cuda"):
    """Run all prompts and return results."""
    results = []

    for i, (ff_val, prompt) in enumerate(PROMPTS):
        prompt_words = prompt.split()
        n_words_to_generate = ff_val - len(prompt_words)

        if n_words_to_generate <= 0:
            n_words_to_generate = ff_val

        print(f"  [{i+1:02d}/53] ff_{ff_val} \"{prompt}\" ({n_words_to_generate} words)...", end="", flush=True)

        text, _, total_ff_words = generate(
            model, n_words_to_generate, prompt,
            temperature=0.8, top_k=50, device=device,
        )

        words = extract_words(text, prompt)

        result = {
            "id": i + 1,
            "ff_target": ff_val,
            "prompt": prompt,
            "prompt_words": len(prompt_words),
            "generated_words": total_ff_words - len(prompt_words),
            "total_words": total_ff_words,
            "error": total_ff_words - ff_val,
            "text": text,
            "first_5": words[:5] if words else [],
            "last_5": words[-5:] if words else [],
        }
        results.append(result)
        print(f" {total_ff_words} words (error: {result['error']:+d})")

    return results


def analyze_results(results: list[dict]):
    """Analyze results and print report."""

    print(f"\n{'='*70}")
    print("RESULTS ANALYSIS")
    print(f"{'='*70}")

    # 1. Length control
    print(f"\n--- 1. LENGTH CONTROL ACCURACY ---\n")

    exact = sum(1 for r in results if r["error"] == 0)
    errors = [abs(r["error"]) for r in results]
    avg_error = sum(errors) / len(errors)

    print(f"  Exact match: {exact}/{len(results)} ({100*exact/len(results):.1f}%)")
    print(f"  Average error: {avg_error:.1f} words")
    print(f"  Maximum error: {max(errors)} words")

    # By range
    faixas = [
        ("Short (50-99)", [r for r in results if 50 <= r["ff_target"] <= 99]),
        ("Medium (100-299)", [r for r in results if 100 <= r["ff_target"] <= 299]),
        ("Long (300-630)", [r for r in results if 300 <= r["ff_target"] <= 630]),
        ("Stress (700-999)", [r for r in results if r["ff_target"] >= 700]),
    ]

    print(f"\n  {'Range':<25} {'Exact':<10} {'Avg error':<12} {'Max error':<10}")
    print(f"  {'-'*57}")
    for nome, grupo in faixas:
        if not grupo:
            continue
        ex = sum(1 for r in grupo if r["error"] == 0)
        errs = [abs(r["error"]) for r in grupo]
        print(f"  {nome:<25} {ex}/{len(grupo):<7} {sum(errs)/len(errs):<12.1f} {max(errs):<10}")

    # 2. Structural planning
    print(f"\n--- 2. STRUCTURAL PLANNING ---\n")

    from collections import Counter
    stopwords = {"o", "a", "os", "as", "de", "do", "da", "dos", "das", "em", "no", "na",
                 "nos", "nas", "e", "ou", "que", "um", "uma", "uns", "umas", "para", "por",
                 "com", "se", "não", "é", "foi", "são", "ao", "à", "aos", "às", "seu", "sua",
                 "seus", "suas", "este", "esta", "esse", "essa", "isso", "isto", "como",
                 "mais", "mas", "também", "já", "ainda", "entre", "sobre", "após", "até",
                 "pelo", "pela", "pelos", "pelas", "ter", "ser", "estar", "há", "tem"}

    inicio_words = Counter()
    fim_words = Counter()

    # Only consider non-stress generations with at least 20 words
    valid = [r for r in results if r["ff_target"] <= 630 and r["generated_words"] >= 20]

    for r in valid:
        words = extract_words(r["text"], r["prompt"])
        first_10 = [w.lower().strip(".,;:!?'\"()[]") for w in words[:10]]
        last_10 = [w.lower().strip(".,;:!?'\"()[]") for w in words[-10:]]

        for w in first_10:
            if w and w not in stopwords and len(w) > 1:
                inicio_words[w] += 1
        for w in last_10:
            if w and w not in stopwords and len(w) > 1:
                fim_words[w] += 1

    print("  Most frequent words at the START (first 10 tokens):")
    for word, count in inicio_words.most_common(15):
        print(f"    {word:<20} {count}")

    print(f"\n  Most frequent words at the END (last 10 tokens):")
    for word, count in fim_words.most_common(15):
        print(f"    {word:<20} {count}")

    # Words extracted from the real dataset (ratio end/start or start/end > 10x)
    # Words that in the training dataset appear almost only at the END of documents
    palavras_de_fim = {
        "grupofolha", "refletir", "tendências", "contemporâneo", "br",
        "editoriais", "assista", "pensamentos", "convite", "enviar",
        "grátis", "completa", "diversas", "tradução", "págs", "anúncios",
        "leia", "trailer", "serão", "anterior", "demais", "lo",
        "disse", "afirmou", "completou", "declarou", "acrescentou",
        "portanto", "assim", "concluiu", "enfim", "encerrou", "final",
        "resultado", "total", "dessa", "desta", "finalizou",
    }
    # Words that in the training dataset appear almost only at the START
    palavras_de_inicio = {
        "anunciou", "decidiu", "nesta", "quinta", "resumo", "empresário",
        "inácio", "fernando", "michel", "joesley", "batista", "aprovou",
        "stf", "feira", "donald", "prendeu", "abriu", "atingiu",
        "saiu", "madrugada", "urgente", "chanceler",
    }

    n_fim_no_fim = sum(fim_words.get(w, 0) for w in palavras_de_fim)
    n_fim_no_inicio = sum(inicio_words.get(w, 0) for w in palavras_de_fim)
    n_inicio_no_inicio = sum(inicio_words.get(w, 0) for w in palavras_de_inicio)
    n_inicio_no_fim = sum(fim_words.get(w, 0) for w in palavras_de_inicio)

    print(f"\n  Typical END words (from dataset):")
    print(f"    Appearances at end of generation:   {n_fim_no_fim}")
    print(f"    Appearances at start of generation: {n_fim_no_inicio}")
    print(f"  Typical START words (from dataset):")
    print(f"    Appearances at start of generation: {n_inicio_no_inicio}")
    print(f"    Appearances at end of generation:   {n_inicio_no_fim}")

    planejamento = (n_fim_no_fim > n_fim_no_inicio) and (n_inicio_no_inicio > n_inicio_no_fim)
    if planejamento:
        print(f"\n  >>> STRUCTURAL PLANNING DETECTED")
        print(f"      Start words at the start, end words at the end — model plans ahead.")
    else:
        print(f"\n  >>> No strong evidence of structural planning")

    # 3. Summary table
    print(f"\n--- 3. SUMMARY TABLE ---\n")
    print(f"  {'ID':<4} {'ff':<6} {'Prompt':<30} {'Generated':<8} {'Error':<6} {'First 3...':<30}")
    print(f"  {'-'*84}")
    for r in results:
        first3 = " ".join(r["first_5"][:3]) if r["first_5"] else "-"
        print(f"  {r['id']:<4} {r['ff_target']:<6} {r['prompt']:<30} {r['generated_words']:<8} {r['error']:+5}  {first3:<30}")

    # 4. Generated texts
    print(f"\n--- 4. GENERATED TEXTS ---\n")
    for r in results:
        print(f"  [{r['id']:02d}] ff_{r['ff_target']} | {r['total_words']} words (error: {r['error']:+d})")
        print(f"  {r['text']}")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    if args.checkpoint is None:
        ckpts = sorted([f for f in glob.glob("checkpoints/model*.pt") if "last" not in f])
        if not ckpts:
            print("No checkpoint found.")
            return
        args.checkpoint = ckpts[-1]

    model = load_checkpoint(args.checkpoint)

    print(f"\nRunning 53 test prompts...")
    print(f"{'-'*70}")
    results = run_prompts(model)

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "test_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_path}")

    # Analysis
    analyze_results(results)

    # Save report as text
    report_path = os.path.join(RESULTS_DIR, "report.txt")
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        analyze_results(results)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
