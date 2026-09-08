r"""
PINN AI Advisor - Script unico per interrogare modelli di frontiera (Claude Opus 5, Sonnet 5, DeepSeek-R1)
sulle problematiche fisiche, matematiche e numeriche del solver in final_roll/src/.

Uso tipico:
  .\venv\Scripts\python tools/pinn_advisor/run_advisor.py --topic "convergenza e bilanciamento loss"
  .\venv\Scripts\python tools/pinn_advisor/run_advisor.py --topic "implementazione VarPro per lambda e mu_p" --model opus
  .\venv\Scripts\python tools/pinn_advisor/run_advisor.py --topic "stiffness singolarita Weissenberg" --model r1
"""

import os
import re
import sys
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Forza encoding UTF-8 su console Windows
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

BASE_DIR = Path(__file__).resolve().parent.parent.parent
ADVISOR_DIR = Path(__file__).resolve().parent
REPORTS_DIR = ADVISOR_DIR / "reports"
ENV_FILE = BASE_DIR / ".env"
PYTHON_EXE = str(BASE_DIR / "venv" / "Scripts" / "python.exe")
KAGGLE_EXE = str(BASE_DIR / "venv" / "Scripts" / "kaggle.exe")

FINAL_ROLL_DIR = BASE_DIR / "final_roll"
SRC_DIR = FINAL_ROLL_DIR / "src"
WIKI_DIR = BASE_DIR / "PINN-wiki"
GEMINI_MD = BASE_DIR / "GEMINI.md"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def read_text_safe(path: Path, max_lines: int = 400) -> str:
    """Legge un file troncandolo se supera max_lines."""
    if not path.exists():
        return f"[File non trovato: {path.name}]"
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        if len(lines) > max_lines:
            return "".join(lines[:max_lines]) + f"\n... [Troncato a {max_lines} righe, totale {len(lines)}] ...\n"
        return "".join(lines)
    except Exception as e:
        return f"[Errore lettura {path.name}: {e}]"

def ensure_kaggle_credentials():
    """Garantisce che il token Kaggle sia disponibile per CLI e Proxy."""
    load_dotenv(ENV_FILE)
    token = os.environ.get("KAGGLE_API_TOKEN")
    if not token:
        user_env_token = os.popen("[System.Environment]::GetEnvironmentVariable('KAGGLE_API_TOKEN', 'User')").read().strip()
        if user_env_token:
            token = user_env_token
            os.environ["KAGGLE_API_TOKEN"] = token
            with open(ENV_FILE, "a", encoding="utf-8") as f:
                f.write(f"\nKAGGLE_API_TOKEN={token}\n")

    access_token_file = Path.home() / ".kaggle" / "access_token"
    if token and not access_token_file.exists():
        access_token_file.parent.mkdir(parents=True, exist_ok=True)
        with open(access_token_file, "w", encoding="utf-8") as f:
            f.write(token)

def refresh_local_proxy():
    """Rinfresca il token del local model proxy se necessario."""
    ensure_kaggle_credentials()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    subprocess.run(
        [KAGGLE_EXE, "benchmarks", "auth", "-y", "--env-file", str(ENV_FILE)],
        cwd=str(BASE_DIR),
        env=env,
        capture_output=True,
        text=True
    )
    load_dotenv(ENV_FILE, override=True)

def build_enriched_prompt(topic: str) -> str:
    """Costruisce il prompt contestualizzato unendo GEMINI.md, Wiki e i componenti chiave di final_roll/src/."""
    # 1. Regole e principi guida da GEMINI.md (estratto essenziale)
    gemini_text = read_text_safe(GEMINI_MD, max_lines=80)

    # 2. Codice essenziale: Physics (residuo momento e costitutivo) e Train (Fase 2)
    physics_code = read_text_safe(SRC_DIR / "physics.py", max_lines=180)
    train_code = read_text_safe(SRC_DIR / "train.py", max_lines=180)

    prompt = f"""Sei uno dei massimi scienziati mondiali specializzati in:
- Physics-Informed Neural Networks (PINN) avanzate
- Fluidodinamica computazionale (CFD) per flussi viscoelastici complessi (Oldroyd-B, PTT, Giesekus)
- Problemi inversi di identificazione parametri e ottimizzazione vincolata

CONTESTO DEL SOLVER (FOUR-ROLL MILL PINN):
1. **Formulazione Stream-Function**: $\\psi$ scalare con divergenza identicamente nulla: $u = \\partial_y \\psi, v = -\\partial_x \\psi$.
2. **Tre Testate Separate**:
   - `model_psi` (1 out) -> cinematica
   - `model_p` (1 out) -> pressione
   - `model_tau` (3 out) -> extra-stress $\\boldsymbol{{\\tau}} = (\\tau_{{xx}}, \\tau_{{xy}}, \\tau_{{yy}})$
3. **Architettura a Fasi Disaccoppiate (Staged Training)**:
   - **Fase 1 (Cinematica & Reologia)**: `model_psi` + `model_tau` addestrati su Oldroyd-B e BC rulli ($w_{{mom}}=0$, pressione disattivata). Identifica $\\lambda$ e $\\mu_p$. Generalmente converge bene.
   - **Fase 2 (Idrodinamica & Pressione)**: `model_tau` è CONGELATO. **`model_psi` RIMANE MOBILE** unitamente a `model_p`. Equazione di conservazione del momento attiva ($w_{{mom}}=1$):
     $$\\rho (\\mathbf{{u}} \\cdot \\nabla) \\mathbf{{u}} = -\\nabla p + \\mu_s \\nabla^2 \\mathbf{{u}} + \\nabla \\cdot \\boldsymbol{{\\tau}}$$
     In questa fase si deve identificare la viscosità del solvente $\\mu_s$ (o $\\mu_{{tot}}$) e ricostruire il campo di pressione $p$.
4. **Condizioni e Dati**: Nessun dato interno di stress da COMSOL. Pressione ancorata in un solo punto ($p(x_0, y_0) = p_{{ref}}$).

ESTRATTI CHIAVE DEL CODICE SORGENTE:
### physics.py (Equazioni di bilancio e momento):
```python
{physics_code}
```

### train.py (Loop di training Fase 2 e gestione gradienti):
```python
{train_code}
```

==============================================================================
OBIETTIVO DELL'ANALISI / QUESITO CRITICO:
==============================================================================
🎯 **{topic.upper()}** 🎯

Mentre la Fase 1 converge adeguatamente, riscontriamo difficoltà nella **Fase 2 (Idrodinamica e Pressione)**, in particolare per:
1. **Identificabilità di $\\mu_s$ (o $\\mu_{{tot}}$) nel problema inverso**: con $\\boldsymbol{{\\tau}}$ congelato e dati di velocità $\\mathbf{{u}}$, qual è la matrice di informazione di Fisher o la sensitività del residuo del momento rispetto a $\\mu_s$ e $\\nabla p$? Esiste accoppiamento spurio / compensazione tra $\\nabla p$ e $\\mu_s \\nabla^2 \\mathbf{{u}}$?
2. **Mobilità di `model_psi` in Fase 2**: Come bilanciare la loss dati di velocità ($W_{{data}}$), il residuo del momento ($W_{{mom}}$) e le BC per evitare che `model_psi` devii dalla cinematica corretta appresa in Fase 1 per "accomodare" artificialmente la loss di Navier-Stokes?
3. **Formulazione dell'equazione del momento**: è preferibile la forma standard in pressione $\\nabla p$, oppure la formulazione in vorticità (curl del momento, eliminando $p$ prima di stimarla), oppure una proiezione variabile (VarPro) per $\\mu_s$?
4. **Proposte concrete e modifiche al codice**: formule matematiche rigorose (KaTeX) e snippet di codice pronti per `final_roll/src/` per sbloccare la convergenza di Fase 2 e garantire l'identificazione esatta di $\\mu_s$.

Rispondi in lingua italiana con rigore accademico e matematico di massimo livello.
"""
    return prompt

def query_local_proxy(model_slug: str, prompt: str) -> str:
    """Esegue la query via local model proxy SDK."""
    refresh_local_proxy()
    import kaggle_benchmarks as kbench
    print(f"⏳ Invocazione modello locale {model_slug} via Kaggle Model Proxy...")
    try:
        llm = kbench.kaggle.load_model(model_slug)
        response = llm.prompt(prompt)
        return response
    except Exception as e:
        print(f"⚠️ Errore con {model_slug}: {e}. Tento refresh credenziali...")
        refresh_local_proxy()
        llm = kbench.kaggle.load_model(model_slug)
        return llm.prompt(prompt)

def query_kaggle_task(model_slug: str, prompt: str, topic_slug: str) -> str:
    """Esegue un benchmark task remoto su Kaggle per modelli frontier (es. Claude Opus 5)."""
    ensure_kaggle_credentials()
    task_name = f"pinn-adv-{topic_slug}"[:30].strip("-")
    task_file = ADVISOR_DIR / f"temp_task_{topic_slug}.py"

    task_code = f'''import kaggle_benchmarks as kbench

PROMPT_TEXT = {repr(prompt)}

@kbench.task(name="{task_name}", description="PINN Advisor Analysis")
def run_analysis(llm) -> dict:
    print("=== INIZIO QUERY ADVISOR ===")
    response = llm.prompt(PROMPT_TEXT)
    print("=== FINE QUERY ===")
    with open("ANALYSIS_RESULT.md", "w", encoding="utf-8") as f:
        f.write(response)
    return {{"result": response}}

if __name__ == "__main__":
    run_analysis.run(kbench.llm)
'''
    with open(task_file, "w", encoding="utf-8") as f:
        f.write(task_code)

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    print(f"📤 Upload task '{task_name}' su Kaggle e attesa build...")
    push_res = subprocess.run(
        [KAGGLE_EXE, "benchmarks", "tasks", "push", task_name, "-f", str(task_file), "--wait"],
        cwd=str(BASE_DIR),
        env=env,
        capture_output=True,
        text=True
    )
    if push_res.returncode != 0:
        print(f"❌ Errore push task:\n{push_res.stderr}\n{push_res.stdout}")
        sys.exit(1)

    print(f"⚡ Esecuzione su Kaggle contro '{model_slug}' (attesa completamento)...")
    run_res = subprocess.run(
        [KAGGLE_EXE, "benchmarks", "tasks", "run", task_name, "-m", model_slug, "--wait"],
        cwd=str(BASE_DIR),
        env=env,
        capture_output=True,
        text=True
    )

    print("📥 Download output del run da Kaggle...")
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    download_dir = ADVISOR_DIR / f"temp_download_{timestamp_str}"
    download_dir.mkdir(exist_ok=True)
    subprocess.run(
        [KAGGLE_EXE, "benchmarks", "tasks", "download", task_name, "-m", model_slug, "-o", str(download_dir), "-f"],
        cwd=str(BASE_DIR),
        env=env,
        capture_output=True,
        text=True
    )

    # Pulizia task temporaneo
    if task_file.exists():
        task_file.unlink(missing_ok=True)

    # Cerca il file markdown scaricato
    content = None
    md_files = list(download_dir.glob("**/ANALYSIS_RESULT.md"))
    if md_files and md_files[0].exists():
        content = md_files[0].read_text(encoding="utf-8", errors="replace")

    # Pulizia cartella download temporanea
    try:
        import shutil
        shutil.rmtree(download_dir, ignore_errors=True)
    except Exception:
        pass

    if content:
        return content

    # Fallback: recupero log se il file markdown non e stato trovato
    log_res = subprocess.run(
        [KAGGLE_EXE, "benchmarks", "tasks", "log", task_name, "-m", model_slug],
        cwd=str(BASE_DIR),
        env=env,
        capture_output=True,
        text=True
    )
    return log_res.stdout if log_res.stdout else "Nessun output recuperato dal task."

def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', '_', text).strip('_')
    return text[:40] if text else "indagine"

def main():
    parser = argparse.ArgumentParser(description="PINN AI Advisor (Four-Roll Mill / final_roll/src)")
    parser.add_argument("-t", "--topic", type=str, required=True,
                        help="Aspetto o domanda da indagare (es. 'convergenza', 'stiffness', 'varpro', 'bilanciamento loss')")
    parser.add_argument("-m", "--model", type=str, default="opus",
                        help="Modello da usare: 'opus' (default: Claude Opus 5), 'sonnet' (Claude Sonnet 5), 'r1' (DeepSeek-R1), 'gemini' (Gemini 3 Flash)")
    parser.add_argument("-r", "--remote", action="store_true",
                        help="Forza esecuzione come benchmark task remoto su Kaggle (default: False, usa proxy locale)")
    args = parser.parse_args()

    topic = args.topic.strip()
    model_choice = args.model.lower().strip()

    # Mappatura modelli
    if model_choice in ["opus", "opus5", "claude-opus", "claude-opus-5-default"]:
        model_slug = "claude-opus-5-default"
        is_remote_task = True  # Opus e disponibile solo come task remoto
    elif model_choice in ["r1", "deepseek", "deepseek-r1", "deepseek-ai/deepseek-r1-0528"]:
        model_slug = "deepseek-ai/deepseek-r1-0528"
        is_remote_task = args.remote
    elif model_choice in ["sonnet", "sonnet5", "fable", "claude-sonnet", "anthropic/claude-sonnet-5@default"]:
        model_slug = "anthropic/claude-sonnet-5@default"
        is_remote_task = args.remote
    elif model_choice in ["gemini", "gemini3", "flash", "google/gemini-3-flash-preview"]:
        model_slug = "google/gemini-3-flash-preview"
        is_remote_task = args.remote
    else:
        model_slug = args.model
        is_remote_task = args.remote

    print("=" * 80)
    print("🔬 PINN AI ADVISOR - FOUR-ROLL MILL")
    print(f"🎯 Aspetto indagato: {topic}")
    print(f"🤖 Modello selezionato: {model_slug} ({'Esecuzione Remota Kaggle' if is_remote_task else 'Model Proxy Locale'})")
    print("=" * 80)

    print("📚 Raccolta contesto da GEMINI.md, Wiki e final_roll/src/...")
    prompt = build_enriched_prompt(topic)

    topic_slug = slugify(topic)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = "opus5" if is_remote_task else model_slug.split("/")[-1].replace("@", "_")
    report_filename = f"REPORT_{timestamp}_{topic_slug}_{model_tag}.md"
    report_path = REPORTS_DIR / report_filename

    if is_remote_task:
        response_text = query_kaggle_task(model_slug, prompt, topic_slug)
    else:
        response_text = query_local_proxy(model_slug, prompt)

    # Scrittura report formattato
    header = f"# Report Analisi PINN: {topic.title()}\n\n"
    header += f"- **Data/Ora:** `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`\n"
    header += f"- **Modello:** `{model_slug}`\n"
    header += f"- **Topic:** `{topic}`\n"
    header += f"- **Target Codebase:** `final_roll/src/`\n\n---\n\n"

    final_report = header + response_text

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(final_report)

    print("\n" + "=" * 80)
    print(f"🎉 Analisi completata con successo!")
    print(f"📄 Report salvato in:\n   👉 {report_path}")
    print("=" * 80 + "\n")

    # Stampa a video
    print(response_text)

if __name__ == "__main__":
    main()
