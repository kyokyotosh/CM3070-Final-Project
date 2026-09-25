"""Audit logged coaching cues for coverage, additions and contradictions.

The coaching layer is judged on two properties (Evaluation chapter):

  coverage      every fault the verdict reports is mentioned in the cue
  faithfulness  the cue neither contradicts the verdict nor adds a correction
                the verdict does not contain

The draft evaluation checked faithfulness for contradicted criteria and
invented faults or numbers. This script applies the full definition to every
logged cue, including unrequested corrections ("engage your core") that the
earlier check did not look for.

Classification is keyword-based, so it is reproducible, and every cue is
written to cue_audit.csv with the matches that triggered each label, so the
automatic labels can be reviewed by hand before any figure is reported.

Faults are derived from the verdict flags (depth_ok, trunk_ok,
knee_travel_ok), which every log schema carries. The two-fault cue cap was
never reached in any logged repetition, so every failed criterion was cued.

Run from backend/:

    python3 audit_cues.py
    python3 audit_cues.py eval_log_v2.csv evaluations/fault_and_cue.csv
"""

import csv
import glob
import os
import re
import sys
from collections import defaultdict

# Each log schema used during the project, identified by its field count.
SCHEMAS = {
    9: ["exercise", "rep", "depth_ok", "trunk_ok", "knee_travel_ok",
        "min_knee", "max_trunk", "max_knee_travel", "coaching"],
    19: ["client_ts", "server_ts", "exercise", "rep", "faults", "faults_cued",
         "depth_ok", "trunk_ok", "knee_travel_ok", "min_knee", "max_trunk",
         "max_knee_travel", "quality", "partial", "analysis_ms", "llm_ms",
         "server_ms", "end_to_end_ms", "coaching"],
    20: ["client_ts", "server_ts", "exercise", "rep", "faults", "faults_cued",
         "depth_ok", "trunk_ok", "knee_travel_ok", "min_knee", "max_trunk",
         "max_knee_travel", "quality", "partial", "analysis_ms", "verdict_ms",
         "queue_ms", "llm_ms", "cue_ms", "coaching"],
    23: ["client_ts", "server_ts", "mode", "exercise", "detected",
         "detect_conf", "rep", "faults", "faults_cued", "depth_ok", "trunk_ok",
         "knee_travel_ok", "min_knee", "max_trunk", "max_knee_travel",
         "quality", "partial", "analysis_ms", "verdict_ms", "queue_ms",
         "llm_ms", "cue_ms", "coaching"],
}

FLAG_TO_FAULT = {"depth_ok": "depth", "trunk_ok": "trunk",
                 "knee_travel_ok": "knee_travel"}

# Words that show a cue mentions a given fault.
COVERAGE = {
    "depth": r"\b(lower|deeper|depth|parallel|bend\w*|down)\b",
    "trunk": r"\b(torso|chest|upright|trunk)\b",
    "knee_travel": r"\b(ankle|toes?)\b",
}

# Correction topics. A match counts as an addition unless the topic belongs
# to a fault the verdict reported.
TOPICS = [
    ("core", r"\bcore\b", None),
    ("back", r"(?<!hips )\bback\b", None),
    ("glutes", r"\bglutes?\b", None),
    ("hips back", r"\bhips back\b", None),
    ("straight line", r"\bstraight line\b|\bhead to heels\b", None),
    ("weight or feet", r"\bweight\b|\bboth feet\b", None),
    ("legs and hips", r"\bengag\w* your (legs|hips)\b", None),
    ("ground, hands or calves", r"\bground\b|\bhands?\b|\bcalves\b", None),
    ("knee alignment",
     r"\bknees? (in line|over (your|the) toes|behind (your|the) toes|"
     r"tracking|to your toes)", "knee_travel"),
    ("torso", r"\b(torso|chest|posture)\b", "trunk"),
    ("depth", r"\b(deeper|parallel)\b", "depth"),
]

# Wording that contradicts a reported fault.
CONTRADICTIONS = {
    "depth": r"\b(good|great|nice|perfect) depth\b|\bextend(ing)?\b",
    "trunk": r"\b(good|great|nice|perfect) posture\b",
    "knee_travel": r"\bknee (is|was) (fine|good)\b",
}
FAULT_PRAISE = r"\bperfect (form|rep)\b"

OUT_PATH = os.path.join("evaluations", "cue_audit.csv")


def parse_file(path):
    rows, skipped = [], 0
    with open(path, newline="") as f:
        for fields in csv.reader(f):
            if not fields or "coaching" in fields:
                continue
            names = SCHEMAS.get(len(fields))
            if names is None:
                skipped += 1
                continue
            row = dict(zip(names, fields))
            if not row.get("coaching"):
                skipped += 1
                continue
            row["source"] = os.path.relpath(path)
            rows.append(row)
    return rows, skipped


def faults_of(row):
    return [fault for flag, fault in FLAG_TO_FAULT.items()
            if row.get(flag) == "False"]


def audit(row):
    text = row["coaching"].lower()
    faults = faults_of(row)

    missing = [f for f in faults if not re.search(COVERAGE[f], text)]

    additions = []
    for name, pattern, owner in TOPICS:
        if owner in faults:
            continue
        if re.search(pattern, text):
            additions.append(name)

    contradictions = [f for f in faults
                      if re.search(CONTRADICTIONS[f], text)]
    if faults and re.search(FAULT_PRAISE, text):
        contradictions.append("praised as perfect")

    return {
        "faults": " ".join(faults),
        "covered": "" if not faults else ("yes" if not missing else
                                          "missing " + " ".join(missing)),
        "additions": "; ".join(additions),
        "contradictions": "; ".join(contradictions),
        "words": len(row["coaching"].split()),
    }


def default_paths():
    paths = glob.glob(os.path.join("evaluations", "*.csv"))
    paths += glob.glob(os.path.join("evaluations", "historical", "*.csv"))
    paths += glob.glob("eval_log*.csv")
    # Never read this script's own output back in as a log.
    return sorted(p for p in set(paths)
                  if os.path.basename(p) != os.path.basename(OUT_PATH))


def pct(n, d):
    return f"{n}/{d} ({100 * n / d:.0f}%)" if d else "0/0"


def main():
    paths = sys.argv[1:] or default_paths()

    seen, results, skipped_total = set(), [], 0
    for path in paths:
        rows, skipped = parse_file(path)
        skipped_total += skipped
        for row in rows:
            key = (row.get("client_ts"), row.get("exercise"), row.get("rep"),
                   row.get("min_knee"), row["coaching"])
            if key in seen:
                continue
            seen.add(key)
            results.append({**row, **audit(row)})

    by_source = defaultdict(list)
    for r in results:
        by_source[r["source"]].append(r)

    print(f"{'source':<44}{'fault cues':>11}{'covered':>14}"
          f"{'additions':>14}{'contradict':>14}")
    fault_all = []
    for source in sorted(by_source):
        fault = [r for r in by_source[source] if r["faults"]]
        fault_all += fault
        if not fault:
            continue
        covered = sum(r["covered"] == "yes" for r in fault)
        added = sum(bool(r["additions"]) for r in fault)
        contra = sum(bool(r["contradictions"]) for r in fault)
        print(f"{source:<44}{len(fault):>11}{pct(covered, len(fault)):>14}"
              f"{pct(added, len(fault)):>14}{pct(contra, len(fault)):>14}")

    n = len(fault_all)
    print(f"\nall fault cues: {n}")
    print(f"  covered        {pct(sum(r['covered'] == 'yes' for r in fault_all), n)}")
    print(f"  with addition  {pct(sum(bool(r['additions']) for r in fault_all), n)}")
    print(f"  contradicting  {pct(sum(bool(r['contradictions']) for r in fault_all), n)}")
    print(f"  over 20 words  {pct(sum(r['words'] > 20 for r in fault_all), n)}")

    topics = defaultdict(int)
    for r in fault_all:
        for t in filter(None, r["additions"].split("; ")):
            topics[t] += 1
    print("\naddition topics in fault cues:")
    for t, c in sorted(topics.items(), key=lambda kv: -kv[1]):
        print(f"  {t:<26}{c}")

    clean = [r for r in results if not r["faults"]]
    print(f"\nclean-rep cues: {len(clean)}")
    print(f"  with addition  {pct(sum(bool(r['additions']) for r in clean), len(clean))}")
    print(f"  say 'perfect'  "
          f"{pct(sum('perfect' in r['coaching'].lower() for r in clean), len(clean))}")

    if skipped_total:
        print(f"\n{skipped_total} rows skipped (unknown schema or no cue)")

    out = OUT_PATH
    fields =["source", "exercise", "rep", "faults", "covered", "additions",
              "contradictions", "words", "coaching"]
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"\nevery cue with its labels written to {out} for manual review")


if __name__ == "__main__":
    main()
