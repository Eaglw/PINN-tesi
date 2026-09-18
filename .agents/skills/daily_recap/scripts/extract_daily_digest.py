#!/usr/bin/env python3
"""
extract_daily_digest.py
Estrae un digest cronologico strutturato di tutte le conversazioni Antigravity,
dei commit Git e delle metriche numeriche registrate durante una determinata giornata.
Utilizzato dalla skill `daily_recap` per compilare il log giornaliero nella Wiki.
"""

import os
import sys
import json
import argparse
import subprocess
from datetime import datetime, date
from pathlib import Path

# Forza stdout in UTF-8 per supportare caratteri speciali su Windows senza crash
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def parse_args():
    parser = argparse.ArgumentParser(description="Estrae il digest delle conversazioni e attivita' giornaliere.")
    parser.add_argument(
        "--date",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Data nel formato YYYY-MM-DD (default: oggi)"
    )
    parser.add_argument(
        "--brain-dir",
        type=str,
        default=os.path.expanduser(r"~\.gemini\antigravity\brain"),
        help="Percorso della directory dei dati di Antigravity (brain)"
    )
    parser.add_argument(
        "--repo-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[4]),
        help="Percorso radice del repository Git"
    )
    return parser.parse_args()

def get_target_conversations(brain_dir: Path, target_date_str: str):
    """Trova tutte le conversazioni con file modificati nella data target."""
    target_d = datetime.strptime(target_date_str, "%Y-%m-%d").date()
    conversations = []
    
    if not brain_dir.exists():
        return conversations

    for conv_dir in brain_dir.iterdir():
        if not conv_dir.is_dir():
            continue
        
        # Controlla la data di modifica della cartella
        mtime = datetime.fromtimestamp(conv_dir.stat().st_mtime).date()
        transcript_path = conv_dir / ".system_generated" / "logs" / "transcript.jsonl"
        
        if transcript_path.exists():
            t_mtime = datetime.fromtimestamp(transcript_path.stat().st_mtime).date()
            if t_mtime == target_d or mtime == target_d:
                conversations.append({
                    "id": conv_dir.name,
                    "path": transcript_path,
                    "mtime": datetime.fromtimestamp(transcript_path.stat().st_mtime)
                })
        elif mtime == target_d:
            conversations.append({
                "id": conv_dir.name,
                "path": None,
                "mtime": datetime.fromtimestamp(conv_dir.stat().st_mtime)
            })

    # Ordina cronologicamente
    conversations.sort(key=lambda x: x["mtime"])
    return conversations

def extract_transcript_events(transcript_path: Path, target_date_str: str):
    """Estrae i prompt utente e le azioni salienti dal file transcript.jsonl."""
    if not transcript_path or not transcript_path.exists():
        return []

    events = []
    try:
        with open(transcript_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                ev_type = obj.get("type", "")
                created_at = obj.get("created_at", "")
                content = obj.get("content", "")
                
                # Filtra per data se creata_at presente
                if created_at and target_date_str not in created_at:
                    continue

                if ev_type == "USER_INPUT":
                    clean_content = content.replace("<USER_REQUEST>", "").replace("</USER_REQUEST>", "").strip()
                    if clean_content:
                        events.append({
                            "type": "USER",
                            "time": created_at[:19] if created_at else "",
                            "text": clean_content
                        })
                elif ev_type == "PLANNER_RESPONSE":
                    tool_calls = obj.get("tool_calls", [])
                    for tc in tool_calls:
                        name = tc.get("name", "")
                        args = tc.get("args", {})
                        if name in ["write_to_file", "replace_file_content"]:
                            target = args.get("TargetFile", "")
                            events.append({
                                "type": "EDIT",
                                "time": created_at[:19] if created_at else "",
                                "text": f"{name}: {Path(target).name if target else ''}"
                            })
                        elif name == "run_command":
                            cmd = args.get("CommandLine", "")
                            if any(k in cmd for k in ["python", "pytest", "git commit", "git push", "comsol"]):
                                events.append({
                                    "type": "CMD",
                                    "time": created_at[:19] if created_at else "",
                                    "text": cmd[:120]
                                })
    except Exception as e:
        events.append({"type": "ERROR", "text": f"Errore lettura transcript: {e}"})

    return events

def get_git_activity(repo_dir: Path, target_date_str: str):
    """Estrae i commit e le statistiche git della giornata."""
    activity = {"commits": [], "stat": ""}
    try:
        # Commit della giornata
        cmd = [
            "git", "log",
            f"--since={target_date_str} 00:00:00",
            f"--until={target_date_str} 23:59:59",
            "--pretty=format:%h - %s (%an, %ar)"
        ]
        res = subprocess.run(cmd, cwd=str(repo_dir), capture_output=True, text=True, errors="replace")
        if res.returncode == 0 and res.stdout.strip():
            activity["commits"] = res.stdout.strip().splitlines()

        # Diff stat sui commit recenti
        cmd_stat = [
            "git", "diff",
            f"--stat", f"HEAD~{min(len(activity['commits']), 5)}", "HEAD"
        ] if activity["commits"] else ["git", "diff", "--stat", "HEAD~1", "HEAD"]
        res_stat = subprocess.run(cmd_stat, cwd=str(repo_dir), capture_output=True, text=True, errors="replace")
        if res_stat.returncode == 0:
            activity["stat"] = res_stat.stdout.strip()
    except Exception as e:
        activity["error"] = str(e)

    return activity

def get_recent_metrics(repo_dir: Path, target_date_str: str):
    """Cerca file metrics_summary.json modificati nella data target."""
    metrics = []
    out_dir = repo_dir / "final_roll" / "output_4rollmill"
    if not out_dir.exists():
        return metrics

    target_d = datetime.strptime(target_date_str, "%Y-%m-%d").date()
    for summary_path in out_dir.rglob("metrics_summary.json"):
        mtime = datetime.fromtimestamp(summary_path.stat().st_mtime).date()
        if mtime == target_d:
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    data["folder"] = summary_path.parent.name
                    metrics.append(data)
            except Exception:
                pass
    return metrics

def main():
    args = parse_args()
    target_date = args.date
    brain_dir = Path(args.brain_dir)
    repo_dir = Path(args.repo_dir)

    print(f"# DAILY DIGEST FOR {target_date}")
    print(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Repository: {repo_dir}")
    print(f"Brain Dir:  {brain_dir}\n")

    # 1. Attivita' Git
    git_act = get_git_activity(repo_dir, target_date)
    print("## Git Commits:")
    if git_act.get("commits"):
        for c in git_act["commits"]:
            print(f"- {c}")
    else:
        print("- Nessun commit registrato con data specifica in questo intervallo.")
    print()

    if git_act.get("stat"):
        print("## Git Diff Stat (File toccati):")
        print("```text")
        print(git_act["stat"])
        print("```\n")

    # 2. Metriche & Benchmark
    metrics = get_recent_metrics(repo_dir, target_date)
    if metrics:
        print("## Benchmark Metrics Trovati:")
        for m in metrics:
            print(f"- Run: `{m.get('folder')}` | Mesh: `{m.get('mesh_tag')}` | lambda: {m.get('lambda_estimated')} (Err: {m.get('lambda_error_pct', 0):.2f}%) | L2_uv: {m.get('l2_uv_mean', 0)*100:.3f}%")
        print()

    # 3. Sessioni Antigravity (Brain Transcripts)
    convs = get_target_conversations(brain_dir, target_date)
    print(f"## Sessioni Antigravity Rilevate ({len(convs)} sessioni):")
    for idx, c in enumerate(convs, 1):
        print(f"\n### Sessione {idx}: `{c['id']}` (Ultima mod: {c['mtime'].strftime('%H:%M:%S')})")
        if c["path"]:
            events = extract_transcript_events(c["path"], target_date)
            user_prompts = [e for e in events if e["type"] == "USER"]
            edits = [e for e in events if e["type"] == "EDIT"]
            cmds = [e for e in events if e["type"] == "CMD"]

            if user_prompts:
                print("**Richieste Utente Chiave:**")
                for u in user_prompts:
                    # Tronca se troppo lungo per brevita'
                    text_preview = u['text'] if len(u['text']) < 250 else u['text'][:250] + "..."
                    print(f"- [{u['time'][11:16]}] {text_preview}")
            
            if edits:
                unique_edits = sorted(list(set(e['text'] for e in edits)))
                print("**File Modificati:** " + ", ".join(f"`{e}`" for e in unique_edits[:10]))

            if cmds:
                unique_cmds = sorted(list(set(e['text'] for e in cmds)))
                print("**Comandi Rilevanti Eseguiti:**")
                for cmd in unique_cmds[:5]:
                    print(f"  - `{cmd}`")
        else:
            print("*(Nessun transcript trovato)*")

    print("\n" + "=" * 80)
    print("Fine Digest.")

if __name__ == "__main__":
    main()
