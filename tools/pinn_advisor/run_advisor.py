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
    subprocess.run(
        [KAGGLE_EXE, "benchmarks", "auth", "-y", "--env-file", str(ENV_FILE)],
        cwd=str(BASE_DIR),
        env=env,
        capture_output=True,
        text=True
    )
    load_dotenv(ENV_FILE, override=True)

def build_enriched_prompt(topic: str) -> str:
    """Costruisce il prompt contestualizzato unendo GEMINI.md, Wiki e final_roll/src/."""
    # 1. Regole e principi guida da GEMINI.md
    gemini_text = read_text_safe(GEMINI_MD, max_lines=150)
    
    # 2. Architettura teorica e lezioni apprese dalla Wiki
    wiki_training = read_text_safe(WIKI_DIR / "Wiki" / "Systems" / "Viscoelastic_Training.md", max_lines=120)

    # 3. Codice sorgente con massimo focus su final_roll/src/
    physics_code = read_text_safe(SRC_DIR / "physics.py", max_lines=350)
    train_code = read_text_safe(SRC_DIR / "train.py", max_lines=320)
    main_code = read_text_safe(FINAL_ROLL_DIR / "train_4roll_main.py", max_lines=180)

    prompt = f"""Sei uno dei massimi scienziati e ricercatori mondiali specializzati in:
- Physics-Informed Neural Networks (PINN) avanzate
- Fluidodinamica computazionale (CFD) per flussi viscoelastici complessi (Oldroyd-B, PTT, Giesekus)
- Ottimizzazione numerica multi-stadio (Adam, L-BFGS, Variable Projection, analisi dell'Hessiana)

Stiamo sviluppando un solver PINN scientifico ad alta precisione per un flusso 2D bidimensionale nel "Four-Roll Mill" (mulino a 4 rulli) che genera un punto di stagnazione iperbolico ad alto tasso estensionale.

==============================================================================
CONTESTO CHIAVE DEL PROGETTO (DA GEMINI.md E DALLA WIKI DI RICERCA)
==============================================================================
1. **Formulazione con Stream-Function**:
   La rete predice scalarmente $\\psi$, imponendo per costruzione la divergenza nulla:
   $$u = \\frac{{\\partial \\psi}}{{\\partial y}}, \\quad v = -\\frac{{\\partial \\psi}}{{\\partial x}} \\implies \\nabla \\cdot \\mathbf{{u}} = 0$$
2. **Tre Testate Distinte (CombinedModel)**:
   - `model_psi` (1 output) per la cinematica
   - `model_p` (1 output) per il campo di pressione
   - `model_tau` (3 output per le componenti simmetriche $\\tau_{{xx}}, \\tau_{{xy}}, \\tau_{{yy}}$)
3. **Strategia a Stadi Disaccoppiati (Decoupled Training)**:
   - **Fase 1 (Cinematica e Reologia)**: addestramento di `model_psi` e `model_tau` sulle equazioni costitutive e sulle condizioni al contorno (velocita imposta e ancoraggio roll stress BC sui rulli). Il momento e spento ($w_{{mom}}=0$), la pressione e congelata. Viene identificato $\\lambda$.
   - **Fase 2 (Idrodinamica e Pressione)**: congeliamo SOLAMENTE `model_tau`. **La stream-function `model_psi` RIMANE MOBILE e trainabile** con $w_{{mom}}=1$ unitamente a `model_p`. Mantenere $\\psi$ mobile e dimostrato essere fondamentale per compensare la componente irrotazionale della velocita e determinare il gradiente di pressione $\\nabla p$.
4. **Regola fondamentale sui dati**:
   Nessun dato di stress interno da COMSOL viene fornito alla rete (tranne l'eventuale ancoraggio BC sui rulli).
5. **Precisione a stadi**: Adam in FP32 $\\rightarrow$ L-BFGS in FP64.

==============================================================================
CODICE SORGENTE CRITICO DI final_roll/ E IN PARTICOLARE final_roll/src/
==============================================================================

### A. final_roll/src/physics.py (PDE Costitutive, Momento, Log-space Parametri):
```python
{physics_code}
```

### B. final_roll/src/train.py (Loop di Addestramento, Closure L-BFGS, Conversioni):
```python
{train_code}
```

### C. final_roll/train_4roll_main.py (Configurazione, Iperparametri, Scaling):
```python
{main_code}
```

==============================================================================
DOMANDA / ASPETTO SPECIFICO DA INDAGARE
==============================================================================
L'utente richiede un'analisi approfondita, rigorosa e focalizzata in particolare sui file della cartella `final_roll/src/` per il seguente aspetto:

🎯 **{topic.upper()}** 🎯

Nel tuo responso:
1. Analizza la formulazione teorica e matematica dell'aspetto richiesto nel contesto del four-roll mill.
2. Esamina criticamente l'implementazione attuale nei file in `final_roll/src/` (`physics.py`, `train.py`) evidenziando colli di bottiglia, incoerenze numeriche o criticita.
3. Fornisci soluzioni e formule matematiche rigorose (in KaTeX/LaTeX).
4. Proponi modifiche concrete al codice (con snippet drop-in ben commentati) destinate a `final_roll/src/`.
5. Rispondi in lingua italiana con tono accademico e scientifico di altissimo livello.
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
    args = parser.parse_args()

    topic = args.topic.strip()
    model_choice = args.model.lower().strip()

    # Mappatura modelli
    if model_choice in ["opus", "opus5", "claude-opus", "claude-opus-5-default"]:
        model_slug = "claude-opus-5-default"
        is_remote_task = True
    elif model_choice in ["r1", "deepseek", "deepseek-r1", "deepseek-ai/deepseek-r1-0528"]:
        model_slug = "deepseek-ai/deepseek-r1-0528"
        is_remote_task = False
    elif model_choice in ["sonnet", "sonnet5", "fable", "claude-sonnet", "anthropic/claude-sonnet-5@default"]:
        model_slug = "anthropic/claude-sonnet-5@default"
        is_remote_task = False
    elif model_choice in ["gemini", "gemini3", "flash", "google/gemini-3-flash-preview"]:
        model_slug = "google/gemini-3-flash-preview"
        is_remote_task = False
    else:
        model_slug = args.model
        is_remote_task = False

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
