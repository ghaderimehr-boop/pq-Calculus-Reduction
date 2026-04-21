"""
sensitivity_analysis.py – Systematic sensitivity analysis of (p,q)-gamma function.
Tests the identity Gamma_{p,q}(z) = Gamma_{q/p}(z) across parameter space.
All core functions are imported from pq_core.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pq_core import q_gamma, pq_gamma   # Import from core (no duplication)

# ----------------------------------------------------------------------
# Systematic sensitivity analysis
# ----------------------------------------------------------------------

def systematic_sensitivity_analysis():
    """Perform comprehensive sensitivity analysis"""
    
    test_cases = [
        {'z': 1.2, 'r': 0.5, 'label': 'Case A'},
        {'z': 2.0, 'r': 0.3, 'label': 'Case B'},
        {'z': 3.1, 'r': 0.7, 'label': 'Case C'},
        {'z': 0.7, 'r': 0.9, 'label': 'Case D'},
        {'z': 4.0, 'r': 0.2, 'label': 'Case E'},
    ]
    
    p_values = np.linspace(0.1, 0.99, 50)
    machine_eps = np.finfo(float).eps
    
    results_table = []
    all_detailed_data = []
    
    print("=" * 80)
    print("SYSTEMATIC SENSITIVITY ANALYSIS")
    print("=" * 80)
    
    for case_idx, case in enumerate(test_cases, 1):
        z = case['z']
        r = case['r']
        
        errors = []
        detailed_case_data = []
        
        gamma_ref = q_gamma(z, r)
        
        for p in p_values:
            q = r * p
            if q >= p or q <= 0:
                continue
            
            gamma_pq = pq_gamma(z, p, q)
            rel_err = abs(gamma_pq - gamma_ref) / abs(gamma_ref)
            errors.append(rel_err)
            
            detailed_case_data.append({
                'case_idx': case_idx,
                'z': z,
                'r': r,
                'p': p,
                'rel_error': rel_err
            })
        
        errors_array = np.array(errors)
        
        case_stats = {
            'z': z,
            'r': r,
            'Δ_mean': np.mean(errors_array),
            'Δ_median': np.median(errors_array),
            'Δ_max': np.max(errors_array),
            'Δ_std': np.std(errors_array),
            'fraction_<ε': np.sum(errors_array < machine_eps) / len(errors_array),
            'fraction_<5ε': np.sum(errors_array < 5 * machine_eps) / len(errors_array),
            'n_samples': len(errors_array),
        }
        
        results_table.append(case_stats)
        all_detailed_data.extend(detailed_case_data)
    
    return results_table, all_detailed_data


# ----------------------------------------------------------------------
# Visualization 
# ----------------------------------------------------------------------

def create_clean_professional_plots(results_table, detailed_data):
    """
    Create clean, publication-quality plots WITHOUT text boxes
    """
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    machine_eps = np.finfo(float).eps
    
    # ----- LEFT PANEL: SENSITIVITY CURVES -----
    cases_to_plot = [1, 2, 3]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o', 's', '^']
    line_styles = ['-', '--', '-.']
    
    for idx, case_num in enumerate(cases_to_plot):
        case_data = [d for d in detailed_data if d['case_idx'] == case_num]
        if not case_data:
            continue
        
        p_vals = [d['p'] for d in case_data]
        errors = [d['rel_error'] for d in case_data]
        
        case_stats = None
        for stats in results_table:
            if stats['z'] == case_data[0]['z'] and stats['r'] == case_data[0]['r']:
                case_stats = stats
                break
        
        if case_stats:
            label = f"$z={case_stats['z']},\\ r={case_stats['r']}$"
        else:
            label = f"Case {case_num}"
        
        ax1.plot(p_vals, errors,
                color=colors[idx],
                linestyle=line_styles[idx],
                marker=markers[idx],
                markersize=4,
                linewidth=1.5,
                alpha=0.8,
                label=label,
                markevery=5)
    
    ax1.axhline(y=machine_eps, color='red', linestyle=':', 
               linewidth=2.0, label='Machine $\\epsilon$', alpha=0.7)
    
    ax1.set_xlabel('Parameter $p$', fontsize=12)
    ax1.set_ylabel('Relative Error $\\Delta(p; z, r)$', fontsize=12)
    ax1.set_title('Sensitivity to $p$ with Fixed $r = q/p$', fontsize=13, pad=15)
    ax1.legend(fontsize=10, loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax1.set_ylim([0, 1.2e-15])
    
    # ----- RIGHT PANEL: ERROR DISTRIBUTION -----
    all_errors = [d['rel_error'] for d in detailed_data]
    all_errors = np.array(all_errors)
    
    log_errors = np.log10(all_errors + 1e-18)
    hist, bin_edges = np.histogram(log_errors, bins=25, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    ax2.bar(bin_centers, hist, width=0.2, alpha=0.7, 
           color='steelblue', edgecolor='black', linewidth=0.5)
    
    ax2.axvline(np.log10(machine_eps), color='red', linestyle='--',
               linewidth=2.0, label=r'$\log_{10}(\epsilon_{\mathrm{mach}})$')
    
    ax2.set_xlabel('$\\log_{10}$(Relative Error $\\Delta$)', fontsize=12)
    ax2.set_ylabel('Probability Density', fontsize=12)
    ax2.set_title('Distribution of Sensitivity Errors', fontsize=13, pad=15)
    ax2.legend(fontsize=10, loc='upper right', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax2.set_xlim([np.min(log_errors) - 0.5, np.max(log_errors) + 0.5])
    
    plt.suptitle('Numerical Sensitivity Analysis of $(p,q)$-Gamma Function', 
                fontsize=14, y=1.02, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('sensitivity_clean_professional.pdf', dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.savefig('sensitivity_clean_professional.png', dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    
    print("\nClean figures saved:")
    print("  - sensitivity_clean_professional.pdf")
    print("  - sensitivity_clean_professional.png")
    
    plt.show()
    return fig


# ----------------------------------------------------------------------
# Generate LaTeX table
# ----------------------------------------------------------------------

def generate_latex_table(results_table):
    """Generate LaTeX code for the sensitivity analysis table"""
    
    latex_code = """\\begin{table}[ht]
\\centering
\\caption{Sensitivity analysis of $\\Gamma_{p,q}(z)$ to parameter $p$ with fixed ratio $r = q/p$. 
$\\Delta_{\\text{mean}}$ and $\\Delta_{\\max}$ denote the mean and maximum relative errors, respectively. 
$\\epsilon_{\\text{mach}} = 2.22\\times10^{-16}$ is the machine epsilon for double precision.}
\\label{tab:sensitivity}
\\begin{tabular}{ccccccccc}
\\toprule
Case & $z$ & $r$ & $\\Delta_{\\text{mean}}$ & $\\Delta_{\\max}$ & $\\sigma(\\Delta)$ & 
\\multicolumn{2}{c}{Fraction of samples} & $n$ \\\\
& & & & & & $<\\epsilon_{\\text{mach}}$ & $<5\\epsilon_{\\text{mach}}$ & \\\\
\\midrule
"""
    
    for i, stats in enumerate(results_table, 1):
        latex_code += f"{i} & {stats['z']} & {stats['r']} & "
        latex_code += f"${stats['Δ_mean']:.2e}$ & ${stats['Δ_max']:.2e}$ & ${stats['Δ_std']:.2e}$ & "
        latex_code += f"{stats['fraction_<ε']*100:.1f}\\% & {stats['fraction_<5ε']*100:.1f}\\% & "
        latex_code += f"{stats['n_samples']}$ \\\\\n"
    
    latex_code += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    return latex_code


# ----------------------------------------------------------------------
# Main execution
# ----------------------------------------------------------------------

def main():
    """Main execution function"""
    
    print("\n" + "="*80)
    print("(p,q)-CALCULUS SENSITIVITY ANALYSIS")
    print("="*80)
    
    print("\n[Step 1/3] Running systematic sensitivity analysis...")
    results_table, detailed_data = systematic_sensitivity_analysis()
    
    print("\n[Step 2/3] Statistical summary:")
    print("-" * 80)
    df = pd.DataFrame(results_table)
    pd.set_option('display.float_format', lambda x: f'{x:.2e}')
    print(df[['z', 'r', 'Δ_mean', 'Δ_max', 'Δ_std', 'fraction_<ε', 'n_samples']])
    
    all_errors = [d['rel_error'] for d in detailed_data]
    print("\n" + "-" * 80)
    print("OVERALL STATISTICS (all cases combined):")
    print(f"  Total samples: {len(all_errors):,}")
    print(f"  Mean relative error: {np.mean(all_errors):.2e}")
    print(f"  Median relative error: {np.median(all_errors):.2e}")
    print(f"  Maximum relative error: {np.max(all_errors):.2e}")
    print(f"  Standard deviation: {np.std(all_errors):.2e}")
    eps = np.finfo(float).eps
    print(f"  Fraction < machine ε: {np.sum(np.array(all_errors) < eps)/len(all_errors)*100:.1f}%")
    
    print("\n[Step 3/3] Creating clean, professional visualization...")
    fig = create_clean_professional_plots(results_table, detailed_data)
    
    print("\n[Additional] Generating LaTeX table code...")
    latex_table = generate_latex_table(results_table)
    
    print("\n" + "="*80)
    print("LATEX TABLE CODE (copy to your .tex file):")
    print("="*80)
    print(latex_table)
    
    with open('sensitivity_table_clean.tex', 'w', encoding='utf-8') as f:
        f.write(latex_table)
    print("\nLaTeX table saved to 'sensitivity_table_clean.tex'")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE SUCCESSFULLY!")
    print("="*80)
    
    return results_table, detailed_data, fig


if __name__ == "__main__":
    results, data, figure = main()
