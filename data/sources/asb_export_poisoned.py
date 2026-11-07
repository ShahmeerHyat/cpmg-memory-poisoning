"""Export ASB's shipped poisoned memory records (Chroma sqlite, opened read-only) to jsonl.

Each record was written by ASB's DPI runs (gpt-4o-mini) in the format of
pyopenagi/agents/react_agent_attack.py:388:
  'Agent: {description}; Task: {task + injected prompt}; Workflow: {workflow}; Tools: {tools}'
We parse the original task, the injection style, and the attacker tool/instruction
(react_agent_attack.py:46-53) so each record can be grouped and paired with clean runs.

Usage: python -I data/sources/asb_export_poisoned.py --raw data/raw/ASB --out data/interim/asb_poisoned.jsonl
"""
import argparse
import glob
import json
import os
import sqlite3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/ASB")
    ap.add_argument("--out", default="data/interim/asb_poisoned.jsonl")
    a = ap.parse_args()
    tools = [json.loads(l) for l in open(os.path.join(a.raw, "data/all_attack_tools.jsonl"))]
    by_instr = {t["Attacker Instruction"]: t for t in tools}
    tasks = [json.loads(l) for l in open(os.path.join(a.raw, "data/agent_task.jsonl"))]
    task_list = [(t["agent_name"], ti, task) for t in tasks for ti, task in enumerate(t["tasks"])]

    out, seen, unparsed = [], set(), 0
    for db in sorted(glob.glob(os.path.join(a.raw, "memory_db/direct_prompt_injection/*/chroma.sqlite3"))):
        style = os.path.basename(os.path.dirname(db)).replace("_gpt-4o-mini", "")
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        rows = con.execute("select e.embedding_id, m.string_value from embeddings e join embedding_metadata m "
                           "on e.id=m.id where m.key='chroma:document' order by e.id").fetchall()
        for emb_id, doc in rows:
            if doc in seen:
                continue
            seen.add(doc)
            tool = next((t for instr, t in by_instr.items() if instr in doc), None)
            task = next(((ag, ti, tk) for ag, ti, tk in task_list if f"Task: {tk}" in doc), None)
            if tool is None or task is None:
                unparsed += 1
            out.append(dict(
                source_dataset="ASB", subset="dpi_memory", source_id=f"asb::{style}::{emb_id}", label=1,
                attack_family=f"asb_{style}", injection_style=style, content=doc.strip(),
                agent=task[0] if task else None, task_idx=task[1] if task else None,
                orig_question=task[2] if task else None,
                attacker_tool=tool["Attacker Tool"] if tool else None,
                attacker_instruction=tool["Attacker Instruction"] if tool else None,
                attack_type=tool["Attack Type"] if tool else None,
                aggressive=tool["Aggressive"] if tool else None,
                record_key=f"asb::{task[0]}::{task[1]}" if task else None,
                generator="gpt-4o-mini (ASB authors)",
                template_ref="pyopenagi/agents/react_agent_attack.py:46-53,388"))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"ASB poisoned unique records: {len(out)}  unparsed (task or tool not matched): {unparsed}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
