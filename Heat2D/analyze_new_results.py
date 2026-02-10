import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import ast

# --- Configurazione Stile ---
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14

def parse_architecture(arch_str):
    """Converte la stringa dell'architettura in una label leggibile."""
    try:
        # Tenta di identificare pattern comuni
        if '50, 50, 50, 50' in arch_str:
            return 'Small (4x50)'
        elif '80, 80, 80, 80, 80, 80' in arch_str:
            return 'Medium (6x80)'
        elif '100, 100, 100, 100, 100, 100, 100, 100' in arch_str:
            return 'Large (8x100)'
        else:
            return 'Custom'
    except:
        return arch_str

def parse_lr_strategy(lr_str):
    """Distingue tra LR fisso e Scheduler."""
    lr_str = str(lr_str)
    if '->' in lr_str:
        return 'Scheduler'
    return 'Fixed (0.001)'

def analyze_results():
    # --- 1. Caricamento Dati ---
    filename = 'results.csv'
    if not os.path.exists(filename):
        print(f"Errore: Il file '{filename}' non è stato trovato.")
        return

    df = pd.read_csv(filename)
    
    # --- 2. Preprocessing ---
    print("Elaborazione dati...")
    
    # Parsing Architettura
    df['Arch_Label'] = df['Architecture'].apply(parse_architecture)
    
    # Ordinamento logico delle dimensioni
    arch_order = ['Small (4x50)', 'Medium (6x80)', 'Large (8x100)']
    
    # Parsing Learning Rate Strategy
    df['LR_Strategy'] = df['Learning_Rate'].apply(parse_lr_strategy)
    
    # Cartella output
    output_dir = 'analysis_plots_new'
    os.makedirs(output_dir, exist_ok=True)

    # --- 3. Generazione Grafici ---

    # ---------------------------------------------------------
    # ANALISI 1: Impatto dell'Architettura
    # ---------------------------------------------------------
    print("Generazione Grafico 1: Architettura...")
    plt.figure(figsize=(16, 8))
    sns.boxplot(data=df, x='Arch_Label', y='L2_Relative_Error', hue='Run_Type', 
                order=arch_order, palette='viridis')
    plt.yscale('log')
    plt.title('Impatto dell\'Architettura sull\'Errore L2 (Scala Log)')
    plt.ylabel('L2 Relative Error (Log)')
    plt.xlabel('Dimensione Architettura')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/1_Architecture_Impact.png')
    plt.close()

    # ---------------------------------------------------------
    # ANALISI 2: Impatto delle Epoche
    # ---------------------------------------------------------
    print("Generazione Grafico 2: Epoche...")
    plt.figure(figsize=(14, 7))
    # Filtriamo per una visualizzazione più pulita, o usiamo tutto
    sns.lineplot(data=df, x='Epochs', y='L2_Relative_Error', hue='Run_Type', 
                 style='Activation_Func', markers=True, dashes=False, err_style="bars", ci=None)
    plt.yscale('log')
    plt.title('Evoluzione dell\'Errore L2 rispetto alle Epoche (20k vs 40k)')
    plt.ylabel('L2 Relative Error (Log)')
    plt.xlabel('Numero di Epoche')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks([20000, 40000]) # Forza a mostrare solo questi tick se sono gli unici valori
    plt.tight_layout()
    plt.savefig(f'{output_dir}/2_Epochs_Impact.png')
    plt.close()

    # ---------------------------------------------------------
    # ANALISI 3: Funzioni di Attivazione
    # ---------------------------------------------------------
    print("Generazione Grafico 3: Funzioni di Attivazione...")
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # L2 Error per Attivazione
    sns.boxplot(data=df, x='Activation_Func', y='L2_Relative_Error', hue='Run_Type', 
                palette='Set2', ax=axes[0])
    axes[0].set_yscale('log')
    axes[0].set_title('L2 Error per Funzione di Attivazione')
    axes[0].legend(loc='lower left', prop={'size': 10})
    
    # Max Peak Error per Attivazione (Importante per vedere instabilità delle PINN)
    sns.boxplot(data=df, x='Activation_Func', y='Max_Relative_Error_Peak', hue='Run_Type', 
                palette='Set2', ax=axes[1])
    axes[1].set_yscale('log')
    axes[1].set_title('Picco Errore Massimo (Stabilità)')
    axes[1].legend([],[], frameon=False) # Rimuovi legenda duplicata
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/3_Activation_Function.png')
    plt.close()

    # ---------------------------------------------------------
    # ANALISI 4: Run Type (NN vs PINN) & Comparazione Generale
    # ---------------------------------------------------------
    print("Generazione Grafico 4: Run Type Overview...")
    plt.figure(figsize=(16, 8))
    
    # Ordiniamo i run type per logica: Prima NN pure, poi PINN
    run_order = ['NN_Random', 'NN_Grid', 'PINN_PurePhys', 'PINN_DataPhys']
    
    sns.violinplot(data=df, x='Run_Type', y='L2_Relative_Error', hue='LR_Strategy',
                   order=run_order, split=True, inner="quart", palette="muted")
    plt.yscale('log')
    plt.title('Confronto Run Type: NN vs PINN (con dettaglio LR Strategy)')
    plt.ylabel('L2 Relative Error (Log)')
    plt.xlabel('Tipo di Run')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/4_RunType_Overview.png')
    plt.close()

    # ---------------------------------------------------------
    # BONUS: LR Scheduler vs Fixed
    # ---------------------------------------------------------
    print("Generazione Grafico Bonus: Scheduler vs Fixed LR...")
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='Run_Type', y='L2_Relative_Error', hue='LR_Strategy', 
                order=run_order, palette='coolwarm', estimator=sum, ci=None) 
    # Nota: Usiamo barplot per vedere l'accumulo o la media
    plt.yscale('log')
    plt.title('Media Errore L2: Learning Rate Fisso vs Scheduler')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/5_LR_Strategy_Impact.png')
    plt.close()

    print(f"\nAnalisi completata! I grafici sono salvati nella cartella: {output_dir}")

if __name__ == "__main__":
    analyze_results()