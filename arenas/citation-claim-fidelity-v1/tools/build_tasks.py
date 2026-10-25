"""Build the citation-claim-fidelity-v1 task set from custody seeds. GOLD-BUILDING code.

Every input and output of this script is real article text or gold, so all of it
lives in article custody, never in this repository:
    $ARTICLE_REPOSITORY/ground_truth/citeclaim-2026-10/build_v1/

Stages (each resumable; results cached per item):

  select    pairs.jsonl -> seeds.jsonl. Deterministic (seeded) choice of citing
            sentences: every source with >=3 citing sentences (capped at 4, for
            the grouping comparison) plus single/double-sentence sources.
  label     Two blind labelers, Claude Opus and Claude Sonnet (hermetic `claude`
            CLI on Claude Max, never the API), label each ORIGINAL sentence
            consistent / contradicts / silent against its abstract. Only
            agreement becomes gold; disagreement is dropped and counted.
  perturb   Claude Opus writes minimal-edit perturbations of each seed both
            labelers called consistent, one per applicable MetaMisCitations
            type (A4 reversal, A2 overgeneralisation, C1 quantitative, C3 method,
            A1 misrepresented finding) plus a faithful paraphrase (a clean trap).
            B3 (wrong paper) is built deterministically: the seed sentence is
            paired with the least-similar abstract in the pool.
            Code-checked: the citation marker survives, the sentence changed,
            and a non-paraphrase edit touches at most MAX_EDIT_TOKENS tokens.
  validate  Claude Sonnet labels every generated item BLIND (same labeling
            prompt, no type shown, fresh call). An item is kept only when the
            blind label is in the intended gold set; drops are counted by type.
  assemble  Items -> tasks: one task = one abstract + one variant of each seed
            that cites it (never two variants of one sentence in a task), so a
            task with k items is a realistic k-citation group.

Conflict of interest: Claude generated and validated the gold, so Claude players
on this arena are flagged in arena.yaml (coi_families) and are not the
benchmark's target (the free models are).

    python arenas/citation-claim-fidelity-v1/tools/build_tasks.py <stage> [--limit N]

Task set v2 (owner decision 2026-10-15: C1 had 11 items, below the recall bar's
pre-registered min_n of 15). v2 = every v1 task byte-for-byte, plus new tasks
built the same way from NEW seeds whose citing sentence and abstract state a
common number, perturbed for C1 only. v1 is fingerprinted and never rebuilt.
Outputs go to build_v2/ (`--build v2`):

  select-c1   pairs.jsonl -> build_v2/seeds.jsonl: pairs not used in v1 whose
              sentence and abstract share a number (citation marker, years and
              the integers 0-2 excluded), at most MAX_SEEDS_PER_SOURCE per source.
  label       as above, on the v2 seeds.
  perturb-c1  Claude Opus writes ONE C1 edit per consistent seed (C1-only prompt).
              Code-checked as above, plus: a number the original shares with the
              abstract no longer appears in the edit.
  validate    as above, on the C1 candidates (blind Sonnet; kept iff 'contradicts').
  assemble-v2 v1 tasks (sha-checked) + new tasks: each agreed ORIG item and each
              validated C1 item, rotated per abstract exactly like v1.

    python arenas/citation-claim-fidelity-v1/tools/build_tasks.py --build v2 <stage> [--limit N]
"""
from __future__ import annotations

import difflib
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from framework import hermetic  # noqa: E402

CUSTODY = Path(os.environ["ARTICLE_REPOSITORY"]) / "ground_truth" / "citeclaim-2026-10"
BUILD_V1 = CUSTODY / "build_v1"
BUILD = BUILD_V1  # `--build v2` points every stage at build_v2/
SEED = 20261014
N_SINGLE_SOURCES = 110
MAX_SEEDS_PER_SOURCE = 4
MAX_EDIT_TOKENS = 8
MAX_PERTURBED_PER_SEED = 2
B3_REUSE = 2
BATCH = 8
TYPES = ("A4", "A2", "C1", "C3", "A1")
#: Blind label(s) that confirm each item type's intended gold.
ACCEPT = {
    "A4": {"contradicts"}, "A2": {"contradicts"}, "C1": {"contradicts"}, "C3": {"contradicts"},
    "A1": {"contradicts", "silent"}, "B3": {"silent", "contradicts"}, "PARA": {"consistent"},
}
#: The gold a player is scored against, per type (evidence-scoped).
GOLD = {
    "A4": "contradicts", "A2": "contradicts", "C1": "contradicts", "C3": "contradicts",
    "A1": "not_consistent", "B3": "not_consistent", "PARA": "consistent",
    "ORIG-consistent": "consistent", "ORIG-silent": "silent", "ORIG-contradicts": "contradicts",
}


def _jl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _append(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _sid(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:12]


# ------------------------------------------------------------------ claude
def claude(model: str, prompt: str, *, timeout: int = 900) -> tuple[str, str | None]:
    """One hermetic Claude CLI call (Claude Max). Returns (text, resolved model id)."""
    argv, kw = hermetic.spawn_kwargs(["claude", "--print", "--model", model, "--output-format", "json"])
    for attempt in range(1, 4):
        cp = subprocess.run(argv, input=prompt, capture_output=True, text=True, encoding="utf-8",
                            errors="replace", timeout=timeout, **kw)
        if cp.returncode == 0:
            try:
                d = json.loads(cp.stdout)
                models = list((d.get("modelUsage") or {}).keys())
                if not d.get("is_error"):
                    return d.get("result") or "", (models[0] if models else None)
            except json.JSONDecodeError:
                pass
        time.sleep(20 * attempt)
    raise RuntimeError(f"claude {model} failed after 3 attempts: {cp.stderr[-400:] or cp.stdout[-400:]}")


def _json_block(text: str):
    s = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", s)
    if m:
        s = m.group(1)
    start = min([i for i in (s.find("["), s.find("{")) if i >= 0], default=-1)
    if start < 0:
        raise ValueError("no JSON in answer")
    # raw_decode reads exactly one JSON value and ignores what follows: a reply
    # with prose (and brackets) after the array crashed the first labeling run.
    value, _end = json.JSONDecoder().raw_decode(s[start:])
    return value


LABEL_PROMPT = """You are building GOLD labels for a benchmark. For each numbered item below you get a cited source's abstract and ONE sentence from a manuscript that cites it. Judge only the claim the sentence attributes to the cited source, using ONLY the abstract as evidence.

Labels:
- "consistent": the abstract supports the claim as stated (faithful paraphrase counts).
- "contradicts": the abstract conflicts with the claim: opposite direction, different number, different design/method/measure, different population, or a claim stated more strongly, more generally or more certainly than the abstract reports.
- "silent": the abstract neither supports nor contradicts it (the claim may still be true in the full paper). Background or methodological uses that the abstract does not address are "silent".

Be strict and literal. If unsure between consistent and silent, choose silent.

Return only a JSON array, one object per item, in order:
[{{"item": "<item id>", "label": "consistent" | "contradicts" | "silent", "evidence_quote": "<shortest verbatim span of the abstract that decides it, or empty for silent>"}}]

ITEMS
{items}"""


def _items_block(rows: list[dict]) -> str:
    out = []
    for r in rows:
        out.append(f"--- item {r['item_id']}\nABSTRACT: {r['abstract']}\nSENTENCE: {r['sentence']}\n")
    return "\n".join(out)


def _label_batch(model: str, rows: list[dict]) -> list[dict]:
    """Label a batch; an unparseable reply is retried once, then the batch is
    labelled one item at a time. An item that still fails gets label None
    (recorded, never guessed, and re-tried on the next run)."""
    for _ in range(2):
        text, resolved = claude(model, LABEL_PROMPT.format(items=_items_block(rows)))
        try:
            arr = _json_block(text)
            if isinstance(arr, list):
                break
        except (ValueError, json.JSONDecodeError):
            pass
    else:
        if len(rows) == 1:
            return [{"item_id": rows[0]["item_id"], "label": None, "evidence_quote": None,
                     "labeler": model, "resolved_model": resolved, "error": "unparseable reply"}]
        return [x for r in rows for x in _label_batch(model, [r])]
    got = {str(a.get("item")): a for a in arr if isinstance(a, dict)}
    out = []
    for r in rows:
        a = got.get(r["item_id"], {})
        lab = a.get("label") if a.get("label") in ("consistent", "contradicts", "silent") else None
        out.append({"item_id": r["item_id"], "label": lab, "evidence_quote": a.get("evidence_quote"),
                    "labeler": model, "resolved_model": resolved})
    return out


# ------------------------------------------------------------------ select
def select() -> None:
    pairs = _jl(CUSTODY / "pairs.jsonl")
    seen, clean = set(), []
    for p in pairs:
        key = (p["cited_doi"], re.sub(r"\s+", " ", p["sentence"]).strip().lower())
        if key in seen:
            continue
        seen.add(key)
        if not (40 <= len(p["sentence"]) <= 450 and 400 <= len(p["abstract"]) <= 3500):
            continue
        clean.append(p)
    by_src = defaultdict(list)
    for p in clean:
        by_src[p["cited_doi"]].append(p)
    rng = random.Random(SEED)
    multi = sorted(d for d, v in by_src.items() if len(v) >= 3)
    single = sorted(d for d, v in by_src.items() if len(v) < 3)
    rng.shuffle(single)
    chosen = []
    for d in multi:
        ps = sorted(by_src[d], key=lambda p: p["pair_id"])
        rng.shuffle(ps)
        chosen += ps[:MAX_SEEDS_PER_SOURCE]
    for d in single[:N_SINGLE_SOURCES]:
        chosen.append(sorted(by_src[d], key=lambda p: p["pair_id"])[0])
    seeds = [{"seed_id": p["pair_id"], **{k: p[k] for k in ("cited_doi", "cited_title", "cited_year",
             "cited_type", "abstract", "sentence", "marker", "prev_sentence", "citing_doi", "section")}}
             for p in chosen]
    BUILD.mkdir(parents=True, exist_ok=True)
    (BUILD / "seeds.jsonl").write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in seeds),
                                       encoding="utf-8")
    print(json.dumps({"pairs": len(pairs), "after_filters": len(clean), "sources_multi": len(multi),
                      "sources_single_used": min(len(single), N_SINGLE_SOURCES), "seeds": len(seeds)}))


# ------------------------------------------------------------------ label
def label(limit: int | None = None) -> None:
    seeds = _jl(BUILD / "seeds.jsonl")[:limit]
    rows = [{"item_id": s["seed_id"], "abstract": s["abstract"], "sentence": s["sentence"]} for s in seeds]
    for model in ("opus", "sonnet"):
        out = BUILD / f"labels_orig_{model}.jsonl"
        done = {r["item_id"] for r in _jl(out) if r.get("label")}
        todo = [r for r in rows if r["item_id"] not in done]
        for i in range(0, len(todo), BATCH):
            _append(out, _label_batch(model, todo[i:i + BATCH]))
            print(f"label {model}: {min(i + BATCH, len(todo))}/{len(todo)}", flush=True)


def _orig_gold() -> dict[str, str]:
    a = {r["item_id"]: r["label"] for r in _jl(BUILD / "labels_orig_opus.jsonl") if r.get("label")}
    b = {r["item_id"]: r["label"] for r in _jl(BUILD / "labels_orig_sonnet.jsonl") if r.get("label")}
    return {k: a[k] for k in a if k in b and a[k] == b[k]}


# ------------------------------------------------------------------ perturb
PERTURB_PROMPT = """You create test items for a benchmark of citation-accuracy checkers. For each numbered seed you get a cited source's abstract and a manuscript sentence that cites it ACCURATELY.

For each seed, write minimal-edit variants of the sentence, one per type below, ONLY where the sentence offers a natural place for that edit. Leave a type out when it does not fit; never force it.

- A4 (reversal): flip the direction or polarity of the finding attributed to the source (increase -> decrease, supported -> did not support, more -> less), so the abstract contradicts it.
- A2 (overgeneralisation): state the attributed finding more broadly or more certainly than the abstract (drop a hedge or qualifier, a specific group -> people in general, "may" -> "does", "some" -> "all", one study -> "research consistently shows"), so the abstract contradicts the stronger claim.
- C1 (quantitative): change a number, percentage, sample size or effect size that the sentence and the abstract both state, so the numbers disagree.
- C3 (method): change the described design, method, measure or population type (experiment -> survey, longitudinal -> cross-sectional, randomized -> observational, children -> adults) so it conflicts with what the abstract reports.
- A1 (misrepresented finding): replace the attributed finding or outcome with a plausible one the source did not report.
- PARA (faithful paraphrase): reword the whole sentence freely while keeping its meaning exactly; it must stay accurate.

Rules for A4, A2, C1, C3, A1: change as few words as possible (at most 6 words), keep everything else identical, and keep the citation marker exactly as it appears. For PARA keep the citation marker exactly too. Do not add explanations to the sentences.

Return only a JSON array:
[{{"seed": "<seed id>", "variants": [{{"type": "A4", "sentence": "..."}}, ...]}}]

SEEDS
{items}"""


def _token_edits(a: str, b: str) -> int:
    ta, tb = a.split(), b.split()
    sm = difflib.SequenceMatcher(a=ta, b=tb, autojunk=False)
    return sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal")


def perturb(limit: int | None = None) -> None:
    gold = _orig_gold()
    seeds = [s for s in _jl(BUILD / "seeds.jsonl") if gold.get(s["seed_id"]) == "consistent"][:limit]
    out = BUILD / "perturbations_raw.jsonl"
    done = {r["seed_id"] for r in _jl(out)}
    todo = [s for s in seeds if s["seed_id"] not in done]
    for i in range(0, len(todo), 5):
        chunk = todo[i:i + 5]
        block = "\n".join(f"--- seed {s['seed_id']}\nABSTRACT: {s['abstract']}\nSENTENCE: {s['sentence']}\n"
                          f"CITATION MARKER: {s['marker']}\n" for s in chunk)
        arr, resolved = [], None
        for _ in range(2):
            text, resolved = claude("opus", PERTURB_PROMPT.format(items=block))
            try:
                arr = _json_block(text)
                if isinstance(arr, list):
                    break
            except (ValueError, json.JSONDecodeError):
                arr = []
        if not isinstance(arr, list):
            arr = []
        got = {str(a.get("seed")): a.get("variants") or [] for a in arr if isinstance(a, dict)}
        _append(out, [{"seed_id": s["seed_id"], "variants": got.get(s["seed_id"], []),
                       "generator": "opus", "resolved_model": resolved} for s in chunk])
        print(f"perturb: {min(i + 5, len(todo))}/{len(todo)}", flush=True)


def _content_words(t: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{4,}", t.lower())}


def candidates() -> list[dict]:
    """Code-checked generated items + deterministic B3 items (not yet validated)."""
    seeds = {s["seed_id"]: s for s in _jl(BUILD / "seeds.jsonl")}
    gold = _orig_gold()
    items, rejects = [], Counter()
    for r in _jl(BUILD / "perturbations_raw.jsonl"):
        s = seeds[r["seed_id"]]
        used = set()
        for v in r["variants"]:
            t, sent = v.get("type"), (v.get("sentence") or "").strip()
            if t not in TYPES + ("PARA",) or t in used:
                rejects[f"{t}:bad_or_duplicate_type"] += 1
                continue
            if not sent or sent == s["sentence"]:
                rejects[f"{t}:unchanged"] += 1
                continue
            if s["marker"] and s["marker"] not in sent:
                rejects[f"{t}:marker_lost"] += 1
                continue
            if t != "PARA" and _token_edits(s["sentence"], sent) > MAX_EDIT_TOKENS:
                rejects[f"{t}:edit_too_large"] += 1
                continue
            used.add(t)
            items.append({"item_id": _sid(s["seed_id"], t), "seed_id": s["seed_id"], "type": t,
                          "sentence": sent, "abstract_doi": s["cited_doi"]})
    # B3 (wrong paper): the seed sentence against the MOST similar other abstract
    # in the pool, a near miss like a real mis-referenced citation. (The first
    # version took the least similar one, which any model spots; consult
    # 2026-10-14.) A near miss can genuinely support the sentence, so blind
    # validation must still call it silent or contradicts. Each abstract serves
    # as a B3 target at most B3_REUSE times; ties break by DOI.
    consistent = sorted(k for k, v in gold.items() if v == "consistent" and k in seeds)
    pool = sorted({s["cited_doi"]: s for s in seeds.values()}.values(), key=lambda s: s["cited_doi"])
    used_as_target: Counter = Counter()
    for k in consistent:
        s = seeds[k]
        cw = _content_words(s["sentence"])
        cands = [p for p in pool if p["cited_doi"] != s["cited_doi"] and used_as_target[p["cited_doi"]] < B3_REUSE
                 and p.get("citing_doi") != s.get("citing_doi")]
        if not cands:
            rejects["B3:no_target"] += 1
            continue
        other = max(cands, key=lambda p: (len(cw & _content_words(p["abstract"])), [-ord(c) for c in p["cited_doi"]]))
        used_as_target[other["cited_doi"]] += 1
        items.append({"item_id": _sid(k, "B3"), "seed_id": k, "type": "B3", "sentence": s["sentence"],
                      "abstract_doi": other["cited_doi"]})
    (BUILD / "candidate_rejects.json").write_text(json.dumps(dict(rejects), indent=1), encoding="utf-8")
    return items


def _abstracts() -> dict[str, dict]:
    return {s["cited_doi"]: s for s in _jl(BUILD / "seeds.jsonl")}


def validate(limit: int | None = None, items_fn=None) -> None:
    abs_by = _abstracts()
    items = (items_fn or candidates)()[:limit]
    out = BUILD / "labels_items_sonnet.jsonl"
    if items_fn is None:  # v1, frozen: unchanged
        done = {r["item_id"] for r in _jl(out) if r.get("label")}
    else:  # v2: a label counts only for the exact sentence it was given
        done = set(_validated_labels(items))
    todo = [{"item_id": it["item_id"], "abstract": abs_by[it["abstract_doi"]]["abstract"],
             "sentence": it["sentence"]} for it in items if it["item_id"] not in done]
    random.Random(SEED).shuffle(todo)  # a batch never lines up a seed's variants side by side
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        sha = {r["item_id"]: _text_sha(r["sentence"]) for r in batch}
        _append(out, [{**lab, "sentence_sha256": sha[lab["item_id"]]} for lab in _label_batch("sonnet", batch)])
        print(f"validate: {min(i + BATCH, len(todo))}/{len(todo)}", flush=True)


# ------------------------------------------------------------------ assemble
HEDGE = re.compile(r"\b(may|might|could|suggest\w*|possibl\w*|likely|appear\w*)\b", re.I)
NEGATIVE = re.compile(r"\b(no (significant )?(effect|difference|association|relationship)|did not|does not|"
                      r"not (significantly|associated|related)|failed to|null)\b", re.I)


def assemble() -> None:
    seeds = {s["seed_id"]: s for s in _jl(BUILD / "seeds.jsonl")}
    abs_by = _abstracts()
    gold_orig = _orig_gold()
    blind = {r["item_id"]: r["label"] for r in _jl(BUILD / "labels_items_sonnet.jsonl") if r.get("label")}
    items, kept, dropped = [], Counter(), Counter()
    for sid, g in gold_orig.items():
        if sid not in seeds:
            continue
        s = seeds[sid]
        traps = [t for t, rx in (("hedged", HEDGE), ("negative_finding", NEGATIVE)) if rx.search(s["sentence"])]
        if s.get("cited_type") == "review":
            traps.append("review")
        items.append({"item_id": _sid(sid, "ORIG"), "seed_id": sid, "type": f"ORIG-{g}",
                      "sentence": s["sentence"], "abstract_doi": s["cited_doi"], "gold": GOLD[f"ORIG-{g}"],
                      "traps": traps})
        kept[f"ORIG-{g}"] += 1
    for sid in seeds:
        if sid not in gold_orig:
            dropped["ORIG-labelers-disagree-or-missing"] += 1
    valid_by_seed = defaultdict(list)
    for it in candidates():
        lab = blind.get(it["item_id"])
        if lab is None:
            dropped[f"{it['type']}:unvalidated"] += 1
            continue
        if lab not in ACCEPT[it["type"]]:
            dropped[f"{it['type']}:blind-label-{lab}"] += 1
            continue
        valid_by_seed[it["seed_id"]].append(it)
    # BALANCE (arena self-test finding, 2026-10-14): with every validated variant
    # kept, perturbed items outnumber clean ones and an always-"contradicts"
    # player scores well on accuracy. So each seed contributes its paraphrase and
    # at most MAX_PERTURBED_PER_SEED perturbed variants, chosen greedily from the
    # globally rarest types so per-type counts stay even. Deterministic.
    type_count: Counter = Counter()
    for sid in sorted(valid_by_seed):
        its = valid_by_seed[sid]
        for it in its:
            if it["type"] == "PARA":
                items.append({**it, "gold": GOLD["PARA"], "traps": []})
                kept["PARA"] += 1
        pert = sorted((it for it in its if it["type"] != "PARA"), key=lambda x: (type_count[x["type"]], x["type"]))
        for it in pert[:MAX_PERTURBED_PER_SEED]:
            items.append({**it, "gold": GOLD[it["type"]], "traps": []})
            kept[it["type"]] += 1
            type_count[it["type"]] += 1
        for it in pert[MAX_PERTURBED_PER_SEED:]:
            dropped[f"{it['type']}:balance-cap"] += 1

    # Tasks: per abstract, one variant of each seed per task, rotating variants.
    by_abs = defaultdict(lambda: defaultdict(list))
    for it in items:
        by_abs[it["abstract_doi"]][it["seed_id"]].append(it)
    rng = random.Random(SEED)
    tasks = []
    for doi in sorted(by_abs):
        per_seed = {sid: sorted(v, key=lambda x: x["item_id"]) for sid, v in by_abs[doi].items()}
        for v in per_seed.values():
            rng.shuffle(v)
        n_tasks = max(len(v) for v in per_seed.values())
        for k in range(n_tasks):
            group = [v[k] for sid, v in sorted(per_seed.items()) if k < len(v)]
            rng.shuffle(group)
            a = abs_by[doi]
            tasks.append({"task_key": _sid(doi, str(k)), "abstract_doi": doi, "title": a["cited_title"],
                          "year": a["cited_year"], "abstract": a["abstract"],
                          "items": [{"item_id": g["item_id"], "sentence": g["sentence"], "type": g["type"],
                                     "gold": g["gold"], "traps": g["traps"], "seed_id": g["seed_id"]}
                                    for g in group]})
    tasks.sort(key=lambda t: t["task_key"])
    out = BUILD / "tasks_v1.jsonl"
    out.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in tasks), encoding="utf-8")
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    sizes = Counter(len(t["items"]) for t in tasks)
    report = {"tasks": len(tasks), "items": sum(len(t["items"]) for t in tasks),
              "group_sizes": dict(sorted(sizes.items())), "kept_by_type": dict(sorted(kept.items())),
              "dropped": dict(sorted(dropped.items())), "tasks_v1_sha256": sha}
    (BUILD / "assemble_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    manifest = CUSTODY / "MANIFEST.json"
    m = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
    m["build_v1/tasks_v1.jsonl"] = {"sha256": sha, "n_tasks": len(tasks), "n_items": report["items"]}
    manifest.write_text(json.dumps(m, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


# ------------------------------------------------------------------ v2 (C1 top-up)
_NUMBER = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(%?)")


def _numbers(text: str, marker: str = "") -> set[str]:
    """Numbers a C1 edit could change: not the citation marker, not a year, not 0-2."""
    out = set()
    for m in _NUMBER.finditer(text):
        v = m.group(1)
        if v == marker or re.fullmatch(r"(19|20)\d\d", v):
            continue
        if "." not in v and int(v) < 3 and not m.group(2):
            continue
        out.add(v)
    return out


def _has_marker(sentence: str, marker: str) -> bool:
    """The citation marker is still there as a whole token ('3' is not found inside '312')."""
    if not marker:
        return True
    edge = r"(?<![0-9A-Za-z])" if marker[0].isalnum() else ""
    tail = r"(?![0-9A-Za-z])" if marker[-1].isalnum() else ""
    return re.search(edge + re.escape(marker) + tail, sentence) is not None


def _text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _validated_labels(items: list[dict]) -> dict[str, str]:
    """item_id -> blind label, only where the label was given for THIS item's sentence.

    Consult 2026-10-15 (Sonnet + Sol): labels keyed by item_id alone would carry a
    'contradicts' earned by one wording to a regenerated one. A label row records
    sentence_sha256 and must match. Rows from before that binding carry no hash;
    they are accepted only while perturbations_raw.jsonl holds one row per seed, so
    each item_id has only ever had one sentence.
    """
    one_text_per_id = max(Counter(r["seed_id"] for r in _jl(BUILD / "perturbations_raw.jsonl")).values(),
                          default=0) <= 1
    rows = {r["item_id"]: r for r in _jl(BUILD / "labels_items_sonnet.jsonl") if r.get("label")}
    out = {}
    for it in items:
        r = rows.get(it["item_id"])
        if r is None:
            continue
        bound = r.get("sentence_sha256")
        if (bound == _text_sha(it["sentence"])) if bound is not None else one_text_per_id:
            out[it["item_id"]] = r["label"]
    return out


def _numbers_only_edit(a: str, b: str) -> bool:
    """Every changed span holds a digit on both sides: the edit changes a number, not
    wording ('45%' -> 'forty-five percent ... disagreed' is not a C1 edit; consult 2026-10-15)."""
    ta, tb = a.split(), b.split()
    ops = [(ta[i1:i2], tb[j1:j2]) for op, i1, i2, j1, j2 in
           difflib.SequenceMatcher(a=ta, b=tb, autojunk=False).get_opcodes() if op != "equal"]
    return bool(ops) and all(any(c.isdigit() for c in " ".join(x)) and any(c.isdigit() for c in " ".join(y))
                             for x, y in ops)


def _norm(sentence: str) -> str:
    return re.sub(r"\s+", " ", sentence).strip().lower()


def select_c1() -> None:
    v1_seeds = _jl(BUILD_V1 / "seeds.jsonl")
    v1_ids = {s["seed_id"] for s in v1_seeds}
    v1_keys = {(s["cited_doi"], _norm(s["sentence"])) for s in v1_seeds}
    pairs = _jl(CUSTODY / "pairs.jsonl")
    seen, clean, why = set(), [], Counter()
    for p in pairs:
        key = (p["cited_doi"], _norm(p["sentence"]))
        if key in seen:
            why["duplicate"] += 1
            continue
        seen.add(key)
        if p["pair_id"] in v1_ids or key in v1_keys:
            why["used_in_v1"] += 1
            continue
        if not (40 <= len(p["sentence"]) <= 450 and 400 <= len(p["abstract"]) <= 3500):
            why["length"] += 1
            continue
        if not (_numbers(p["sentence"], p.get("marker") or "") & _numbers(p["abstract"])):
            why["no_shared_number"] += 1
            continue
        clean.append(p)
    by_src = defaultdict(list)
    for p in clean:
        by_src[p["cited_doi"]].append(p)
    rng = random.Random(SEED + 2)
    chosen = []
    for d in sorted(by_src):
        ps = sorted(by_src[d], key=lambda p: p["pair_id"])
        rng.shuffle(ps)
        chosen += ps[:MAX_SEEDS_PER_SOURCE]
        why["source_cap"] += max(0, len(ps) - MAX_SEEDS_PER_SOURCE)
    seeds = [{"seed_id": p["pair_id"], **{k: p[k] for k in ("cited_doi", "cited_title", "cited_year",
             "cited_type", "abstract", "sentence", "marker", "prev_sentence", "citing_doi", "section")}}
             for p in chosen]
    BUILD.mkdir(parents=True, exist_ok=True)
    (BUILD / "seeds.jsonl").write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in seeds),
                                       encoding="utf-8")
    print(json.dumps({"pairs": len(pairs), "excluded": dict(why), "sources": len(by_src), "seeds": len(seeds)}))


PERTURB_C1_PROMPT = """You create test items for a benchmark of citation-accuracy checkers. For each numbered seed you get a cited source's abstract and a manuscript sentence that cites it ACCURATELY.

For each seed, write ONE minimal-edit variant of the sentence of this type, ONLY where the sentence offers a natural place for it. Leave the seed out when it does not fit; never force it.

- C1 (quantitative): change a number, percentage, sample size or effect size that the sentence and the abstract both state, so the numbers disagree.

Rules: change as few words as possible (at most 6 words), keep everything else identical, and keep the citation marker exactly as it appears. Do not add explanations to the sentences.

Return only a JSON array:
[{{"seed": "<seed id>", "variants": [{{"type": "C1", "sentence": "..."}}]}}]

SEEDS
{items}"""


def perturb_c1(limit: int | None = None) -> None:
    gold = _orig_gold()
    seeds = [s for s in _jl(BUILD / "seeds.jsonl") if gold.get(s["seed_id"]) == "consistent"][:limit]
    out = BUILD / "perturbations_raw.jsonl"
    done = {r["seed_id"] for r in _jl(out)}
    todo = [s for s in seeds if s["seed_id"] not in done]
    for i in range(0, len(todo), 5):
        chunk = todo[i:i + 5]
        block = "\n".join(f"--- seed {s['seed_id']}\nABSTRACT: {s['abstract']}\nSENTENCE: {s['sentence']}\n"
                          f"CITATION MARKER: {s['marker']}\n" for s in chunk)
        arr, resolved = [], None
        for _ in range(2):
            text, resolved = claude("opus", PERTURB_C1_PROMPT.format(items=block))
            try:
                arr = _json_block(text)
                if isinstance(arr, list):
                    break
            except (ValueError, json.JSONDecodeError):
                arr = []
        if not isinstance(arr, list):
            arr = []
        got = {str(a.get("seed")): a.get("variants") or [] for a in arr if isinstance(a, dict)}
        _append(out, [{"seed_id": s["seed_id"], "variants": got.get(s["seed_id"], []),
                       "generator": "opus", "prompt": "PERTURB_C1_PROMPT", "resolved_model": resolved}
                      for s in chunk])
        print(f"perturb-c1: {min(i + 5, len(todo))}/{len(todo)}", flush=True)


def candidates_c1() -> list[dict]:
    """Code-checked C1 items of the v2 seeds (not yet validated). No B3, no other types."""
    seeds = {s["seed_id"]: s for s in _jl(BUILD / "seeds.jsonl")}
    items, rejects = [], Counter()
    for r in _jl(BUILD / "perturbations_raw.jsonl"):
        s = seeds[r["seed_id"]]
        shared = _numbers(s["sentence"], s["marker"] or "") & _numbers(s["abstract"])
        for v in [v for v in r["variants"] if v.get("type") == "C1"][:1]:
            sent = (v.get("sentence") or "").strip()
            if not sent or sent == s["sentence"]:
                rejects["C1:unchanged"] += 1
            elif not _has_marker(sent, s["marker"] or ""):
                rejects["C1:marker_lost"] += 1
            elif _token_edits(s["sentence"], sent) > MAX_EDIT_TOKENS:
                rejects["C1:edit_too_large"] += 1
            elif not (shared - _numbers(sent, s["marker"] or "")):
                rejects["C1:no_shared_number_changed"] += 1
            elif not _numbers_only_edit(s["sentence"], sent):
                rejects["C1:non_numeric_edit"] += 1
            else:
                items.append({"item_id": _sid(s["seed_id"], "C1"), "seed_id": s["seed_id"], "type": "C1",
                              "sentence": sent, "abstract_doi": s["cited_doi"]})
    (BUILD / "candidate_rejects.json").write_text(json.dumps(dict(rejects), indent=1), encoding="utf-8")
    return items


def assemble_v2() -> None:
    manifest = CUSTODY / "MANIFEST.json"
    m = json.loads(manifest.read_text(encoding="utf-8"))
    v1_path = BUILD_V1 / "tasks_v1.jsonl"
    v1_sha = hashlib.sha256(v1_path.read_bytes()).hexdigest()
    if v1_sha != m["build_v1/tasks_v1.jsonl"]["sha256"]:
        raise SystemExit(f"tasks_v1.jsonl sha256 {v1_sha[:12]} != MANIFEST; refusing to build v2 on altered v1")
    v1_tasks = _jl(v1_path)
    seeds = {s["seed_id"]: s for s in _jl(BUILD / "seeds.jsonl")}
    gold_orig = _orig_gold()
    c1_candidates = candidates_c1()
    blind = _validated_labels(c1_candidates)
    items, kept, dropped = [], Counter(), Counter()
    for sid, g in sorted(gold_orig.items()):
        s = seeds[sid]
        traps = [t for t, rx in (("hedged", HEDGE), ("negative_finding", NEGATIVE)) if rx.search(s["sentence"])]
        if s.get("cited_type") == "review":
            traps.append("review")
        items.append({"item_id": _sid(sid, "ORIG"), "seed_id": sid, "type": f"ORIG-{g}",
                      "sentence": s["sentence"], "abstract_doi": s["cited_doi"], "gold": GOLD[f"ORIG-{g}"],
                      "traps": traps})
        kept[f"ORIG-{g}"] += 1
    dropped["ORIG-labelers-disagree-or-missing"] = sum(1 for sid in seeds if sid not in gold_orig)
    for it in c1_candidates:
        lab = blind.get(it["item_id"])
        if lab is None:
            dropped["C1:unvalidated"] += 1
        elif lab not in ACCEPT["C1"]:
            dropped[f"C1:blind-label-{lab}"] += 1
        else:
            items.append({**it, "gold": GOLD["C1"], "traps": []})
            kept["C1"] += 1
    by_abs = defaultdict(lambda: defaultdict(list))
    for it in items:
        by_abs[it["abstract_doi"]][it["seed_id"]].append(it)
    rng = random.Random(SEED + 2)
    new_tasks = []
    for doi in sorted(by_abs):
        per_seed = {sid: sorted(v, key=lambda x: x["item_id"]) for sid, v in by_abs[doi].items()}
        for v in per_seed.values():
            rng.shuffle(v)
        for k in range(max(len(v) for v in per_seed.values())):
            group = [v[k] for sid, v in sorted(per_seed.items()) if k < len(v)]
            rng.shuffle(group)
            a = seeds[group[0]["seed_id"]]
            new_tasks.append({"task_key": _sid(doi, "v2", str(k)), "abstract_doi": doi, "title": a["cited_title"],
                              "year": a["cited_year"], "abstract": a["abstract"],
                              "items": [{"item_id": g["item_id"], "sentence": g["sentence"], "type": g["type"],
                                         "gold": g["gold"], "traps": g["traps"], "seed_id": g["seed_id"]}
                                        for g in group]})
    clash = {t["task_key"] for t in v1_tasks} & {t["task_key"] for t in new_tasks}
    if clash:
        raise SystemExit(f"{len(clash)} v2 task keys collide with v1")
    tasks = sorted(v1_tasks + new_tasks, key=lambda t: t["task_key"])
    out = BUILD / "tasks_v2.jsonl"
    out.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in tasks), encoding="utf-8")
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    all_types = Counter(it["type"] for t in tasks for it in t["items"])
    report = {"tasks": len(tasks), "tasks_from_v1": len(v1_tasks), "tasks_new": len(new_tasks),
              "items": sum(len(t["items"]) for t in tasks),
              "items_new": sum(len(t["items"]) for t in new_tasks),
              "group_sizes_new": dict(sorted(Counter(len(t["items"]) for t in new_tasks).items())),
              "kept_new_by_type": dict(sorted(kept.items())), "dropped_new": dict(sorted(dropped.items())),
              "items_by_type_v2": dict(sorted(all_types.items())), "tasks_v1_sha256": v1_sha,
              "tasks_v2_sha256": sha}
    (BUILD / "assemble_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    m["build_v2/tasks_v2.jsonl"] = {"sha256": sha, "n_tasks": len(tasks), "n_items": report["items"]}
    manifest.write_text(json.dumps(m, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--build" in args:
        i = args.index("--build")
        if args[i + 1] != "v2":
            raise SystemExit("--build takes v2 (v1 is the default)")
        BUILD = CUSTODY / "build_v2"
        del args[i:i + 2]
    stage = args[0]
    lim = int(args[args.index("--limit") + 1]) if "--limit" in args else None
    v2 = BUILD.name == "build_v2"
    stages = {"select": select, "label": lambda: label(lim), "perturb": lambda: perturb(lim),
              "validate": lambda: validate(lim), "assemble": assemble} if not v2 else {
        "select-c1": select_c1, "label": lambda: label(lim), "perturb-c1": lambda: perturb_c1(lim),
        "validate": lambda: validate(lim, candidates_c1), "assemble-v2": assemble_v2}
    if stage not in stages:
        raise SystemExit(f"stage {stage!r} not available for {BUILD.name}: {sorted(stages)}")
    stages[stage]()
