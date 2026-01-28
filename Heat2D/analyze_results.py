import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Configurazione stile
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (16, 8)

def analyze_results_final_v5():
    # --- Caricamento Dati ---
    try:
        df = pd.read_csv('results.csv')
    except FileNotFoundError:
        print("Errore: File 'results.csv' non trovato nella directory corrente.")
        return

    # --- Creazione Cartella Output ---
    output_dir = 'analysis_plots'
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Directory '{output_dir}' pronta.")
    except Exception as e:
        print(f"Errore nella creazione della directory: {e}")
        return

    # --- Preprocessing & Utility ---
    def get_arch_label(x):
        if '50, 50, 50' in x: return 'Small (4x50)'
        if '80, 80, 80' in x: return 'Large (6x80)'
        return 'Other'

    df['Arch_Label'] = df['Architecture'].apply(get_arch_label)

    # Definizione Categorie Base
    def get_category_base(row):
        rtype = row['Run_Type']
        if 'PINN' in rtype:
            return rtype
        pts = row['n_points']
        # Raggruppa 500 e 506 sotto '500'
        pt_label = '2000' if pts == 2000 else '500' 
        base = 'NN_Grid' if 'Grid' in rtype else 'NN_Rand'
        return f"{base}{pt_label}"

    df['Category'] = df.apply(get_category_base, axis=1)

    # Definizione Categorie Weighted
    def get_category_weighted(row):
        base = get_category_base(row)
        if row['Loss_Weight'] != 'not_weighted':
            return base + "_Weighted"
        return base

    df['Category_Weighted'] = df.apply(get_category_weighted, axis=1)
    
    # Ordini fissi per coerenza visiva
    order_cats_p2 = ['NN_Grid2000', 'NN_Rand2000', 'NN_Grid500', 'NN_Rand500', 'PINN_DataPhys', 'PINN_PurePhys']
    
    order_cats_p3 = ['NN_Grid2000', 'NN_Rand2000', 'NN_Grid500', 'NN_Rand500', 
                     'PINN_DataPhys', 'PINN_PurePhys', 
                     'PINN_DataPhys_Weighted', 'PINN_PurePhys_Weighted']

    # ================= FASE 1 =================
    print("Generazione FASE 1...")
    df_p1 = df[df['n_points'] == 2000].copy()
    
    fig1, axes1 = plt.subplots(1, 2, figsize=(16, 6))
    
    # L2 Error (Qui hue è diverso da x, quindi non serve correzione, è standard)
    sns.boxplot(data=df_p1, x='Run_Type', y='L2_Relative_Error', hue='Activation_Func', showmeans=True, ax=axes1[0])
    axes1[0].set_yscale('log')
    axes1[0].set_title('Phase 1: L2 Error Distribution (n=2000)')
    
    # Max Error
    sns.boxplot(data=df_p1, x='Run_Type', y='Max_Relative_Error_Peak', hue='Activation_Func', showmeans=True, ax=axes1[1])
    axes1[1].set_yscale('log')
    axes1[1].set_title('Phase 1: Max Peak Error Distribution (n=2000)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'phase1_analysis.png'))

    # ================= FASE 2 =================
    print("Generazione FASE 2 (Fix Warning)...")
    df_p2 = df[df['n_points'] != 1000].copy()
    
    fig2, axes2 = plt.subplots(1, 2, figsize=(18, 8))
    
    # FIX APPLICATO: Aggiunto hue='Category' e legend=False
    # L2 Error
    sns.boxplot(data=df_p2, x='Category', y='L2_Relative_Error', hue='Category', legend=False,
                showmeans=True, order=order_cats_p2, palette='viridis', ax=axes2[0])
    axes2[0].set_yscale('log')
    axes2[0].set_title('Phase 2: L2 Error Distribution by Category')
    axes2[0].tick_params(axis='x', rotation=45)

    # Max Error
    sns.boxplot(data=df_p2, x='Category', y='Max_Relative_Error_Peak', hue='Category', legend=False,
                showmeans=True, order=order_cats_p2, palette='viridis', ax=axes2[1])
    axes2[1].set_yscale('log')
    axes2[1].set_title('Phase 2: Max Error Distribution by Category')
    axes2[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'phase2_analysis.png'))

    # ================= FASE 3 (Panoramica Generale) =================
    print("Generazione FASE 3 Overview (Fix Warning)...")
    df_p3 = df.copy()
    plot_df_p3 = df_p3[df_p3['Category_Weighted'].isin(order_cats_p3)].copy()
    
    fig3, axes3 = plt.subplots(1, 2, figsize=(18, 8))
    
    # FIX APPLICATO: Aggiunto hue='Category_Weighted' e legend=False
    # L2 Error
    sns.boxplot(data=plot_df_p3, x='Category_Weighted', y='L2_Relative_Error', hue='Category_Weighted', legend=False,
                showmeans=True, order=order_cats_p3, palette='magma', ax=axes3[0])
    axes3[0].set_yscale('log')
    axes3[0].set_title('Phase 3: Impact of Weighting (Overview)')
    axes3[0].tick_params(axis='x', rotation=45)
    
    # Max Error
    sns.boxplot(data=plot_df_p3, x='Category_Weighted', y='Max_Relative_Error_Peak', hue='Category_Weighted', legend=False,
                showmeans=True, order=order_cats_p3, palette='magma', ax=axes3[1])
    axes3[1].set_yscale('log')
    axes3[1].set_title('Phase 3: Impact of Weighting (Overview)')
    axes3[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'phase3_overview.png'))

    # ================= FASE 3 (Deep Dive PINN) =================
    print("Generazione FASE 3 Deep Dive...")
    pinn_df = df[df['Run_Type'].str.contains('PINN')].copy()
    
    # Creazione etichetta asse X più leggibile
    pinn_df['Weight_Status'] = pinn_df['Loss_Weight'].apply(lambda x: 'Weighted' if x != 'not_weighted' else 'Not Weighted')
    
    fig4, axes4 = plt.subplots(1, 2, figsize=(16, 6))
    
    # Qui hue è Activation_Func (diverso da x), quindi sintassi standard corretta
    # L2 Deep Dive
    sns.boxplot(data=pinn_df, x='Weight_Status', y='L2_Relative_Error', hue='Activation_Func', showmeans=True, ax=axes4[0])
    axes4[0].set_yscale('log')
    axes4[0].set_title('PINN Detail: Weighted vs Unweighted Impact (L2)')
    
    # Max Error Deep Dive
    sns.boxplot(data=pinn_df, x='Weight_Status', y='Max_Relative_Error_Peak', hue='Activation_Func', showmeans=True, ax=axes4[1])
    axes4[1].set_yscale('log')
    axes4[1].set_title('PINN Detail: Weighted vs Unweighted Impact (Max Error)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'phase3_deep_dive.png'))
    
    print(f"Fatto! Tutti i grafici corretti sono stati salvati nella cartella '{output_dir}'.")

if __name__ == "__main__":
    analyze_results_final_v5()