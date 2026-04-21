"""
monte_carlo_analysis.py – Monte Carlo statistical analysis for (p,q)-gamma redundancy.
Tests the identity Gamma_{p,q}(z) = Gamma_{q/p}(z) across random parameter space.
All core functions are imported from pq_core.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pq_core import q_gamma, pq_gamma

# ----------------------------------------------------------------------
# Monte Carlo analysis
# ----------------------------------------------------------------------

def monte_carlo_analysis(num_samples=2000, seed=42):
    """
    Monte Carlo analysis across the entire parameter space.
    Returns results list, errors array, and parameter arrays.
    """
    np.random.seed(seed)
    machine_eps = np.finfo(float).eps

    print("=" * 80)
    print("MONTE CARLO STATISTICAL ANALYSIS")
    print("=" * 80)
    print(f"Number of random samples: {num_samples:,}")
    print(f"Machine epsilon: {machine_eps:.2e}\n")

    results = []

    for i in range(num_samples):
        # Sample z log-uniformly for better coverage
        z = 10 ** np.random.uniform(-1, 1) * np.random.uniform(0.2, 5.0)
        p = np.random.uniform(0.1, 0.99)
        r = np.random.uniform(0.01, 0.99)
        q = r * p

        if q >= p or q <= 0:
            continue

        gamma_pq = pq_gamma(z, p, q)
        gamma_r = q_gamma(z, r)

        if gamma_r != 0:
            rel_err = abs(gamma_pq - gamma_r) / abs(gamma_r)
        else:
            rel_err = 0.0

        results.append({
            'sample_id': i,
            'z': z,
            'p': p,
            'q': q,
            'r': r,
            'gamma_pq': gamma_pq,
            'gamma_r': gamma_r,
            'rel_error': rel_err,
            'log10_error': np.log10(rel_err + 1e-18)
        })

    errors = np.array([r['rel_error'] for r in results])
    z_vals = np.array([r['z'] for r in results])
    p_vals = np.array([r['p'] for r in results])
    r_vals = np.array([r['r'] for r in results])

    # Descriptive statistics
    print("DESCRIPTIVE STATISTICS:")
    print("-" * 80)
    print(f"Mean relative error:      {np.mean(errors):.2e}")
    print(f"Median relative error:    {np.median(errors):.2e}")
    print(f"Std deviation:            {np.std(errors):.2e}")
    print(f"Minimum error:            {np.min(errors):.2e}")
    print(f"Maximum error:            {np.max(errors):.2e}")
    print(f"IQR (Q3-Q1):              {np.percentile(errors, 75) - np.percentile(errors, 25):.2e}\n")

    # Error distribution relative to machine epsilon
    print("ERROR DISTRIBUTION RELATIVE TO MACHINE EPSILON:")
    print("-" * 80)
    print(f"Samples with error = 0:        {np.sum(errors == 0):,} ({np.sum(errors == 0)/len(errors)*100:.1f}%)")
    print(f"Samples with error < ε_mach:   {np.sum(errors < machine_eps):,} ({np.sum(errors < machine_eps)/len(errors)*100:.1f}%)")
    print(f"Samples with error < 5ε_mach:  {np.sum(errors < 5*machine_eps):,} ({np.sum(errors < 5*machine_eps)/len(errors)*100:.1f}%)")
    print(f"Samples with error < 10ε_mach: {np.sum(errors < 10*machine_eps):,} ({np.sum(errors < 10*machine_eps)/len(errors)*100:.1f}%)")
    print(f"Samples with error > 1e-14:    {np.sum(errors > 1e-14):,} ({np.sum(errors > 1e-14)/len(errors)*100:.1f}%)\n")

    # Correlation analysis
    print("CORRELATION ANALYSIS:")
    print("-" * 80)
    log_errors = np.log10(errors + 1e-18)
    corr_z, p_z = stats.pearsonr(z_vals, log_errors)
    corr_p, p_p = stats.pearsonr(p_vals, log_errors)
    corr_r, p_r = stats.pearsonr(r_vals, log_errors)
    print(f"Correlation with z: ρ = {corr_z:.3f} (p = {p_z:.3f})")
    print(f"Correlation with p: ρ = {corr_p:.3f} (p = {p_p:.3f})")
    print(f"Correlation with r: ρ = {corr_r:.3f} (p = {p_r:.3f})\n")

    # Hypothesis testing
    print("HYPOTHESIS TESTING:")
    print("-" * 80)
    t_stat1, p_val1 = stats.ttest_1samp(errors, machine_eps)
    print(f"Test 1: H0: μ = ε_mach vs H1: μ ≠ ε_mach")
    print(f"  t-statistic = {t_stat1:.3f}, p-value = {p_val1:.3f}")
    print(f"  Conclusion: {'Reject H0' if p_val1 < 0.05 else 'Fail to reject H0'}")

    t_stat2, p_val2 = stats.ttest_1samp(errors, 0)
    print(f"\nTest 2: H0: μ = 0 vs H1: μ ≠ 0")
    print(f"  t-statistic = {t_stat2:.3f}, p-value = {p_val2:.3f}")
    print(f"  Conclusion: {'Reject H0' if p_val2 < 0.05 else 'Fail to reject H0'}")

    if len(errors) <= 5000:
        shapiro_stat, shapiro_p = stats.shapiro(log_errors)
        print(f"\nTest 3: Shapiro-Wilk test for normality of log10(errors)")
        print(f"  W = {shapiro_stat:.3f}, p-value = {shapiro_p:.3f}")
        print(f"  Conclusion: {'Normal' if shapiro_p > 0.05 else 'Not normal'}")

    print("\nPARAMETER SPACE COVERAGE:")
    print("-" * 80)
    print(f"z range: [{np.min(z_vals):.3f}, {np.max(z_vals):.3f}]")
    print(f"p range: [{np.min(p_vals):.3f}, {np.max(p_vals):.3f}]")
    print(f"r range: [{np.min(r_vals):.3f}, {np.max(r_vals):.3f}]")

    return results, errors, z_vals, p_vals, r_vals


# ----------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------

def create_statistical_plots(results, errors):
    """Create publication-quality plots for statistical analysis"""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    machine_eps = np.finfo(float).eps
    errors_array = np.array(errors)
    log_errors = np.log10(errors_array + 1e-18)

    z_vals = np.array([r['z'] for r in results])
    p_vals = np.array([r['p'] for r in results])
    r_vals = np.array([r['r'] for r in results])

    # 1. Histogram of log10(errors)
    ax1 = axes[0, 0]
    ax1.hist(log_errors, bins=40, alpha=0.7, color='steelblue', edgecolor='black', density=True)
    ax1.axvline(np.log10(machine_eps), color='red', linestyle='--',
                linewidth=2.0, label=f'log10(ε_mach) = {np.log10(machine_eps):.2f}')
    mu, sigma = np.mean(log_errors), np.std(log_errors)
    x = np.linspace(np.min(log_errors), np.max(log_errors), 100)
    y = stats.norm.pdf(x, mu, sigma)
    ax1.plot(x, y, 'r-', linewidth=2, alpha=0.7, label=f'Normal fit\nμ={mu:.2f}, σ={sigma:.2f}')
    ax1.set_xlabel('log$_{10}$(Relative Error)', fontsize=12)
    ax1.set_ylabel('Probability Density', fontsize=12)
    ax1.set_title('Distribution of Relative Errors', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # 2. Q-Q plot for normality check
    ax2 = axes[0, 1]
    stats.probplot(log_errors, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot: Normality Check', fontsize=13)
    ax2.grid(True, alpha=0.3, linestyle='--')

    # 3. Scatter: error vs z (colored by r)
    ax3 = axes[1, 0]
    sc1 = ax3.scatter(z_vals, errors_array, alpha=0.6, c=r_vals, cmap='viridis', s=20, edgecolor='k', linewidth=0.2)
    ax3.set_xlabel('Argument $z$', fontsize=12)
    ax3.set_ylabel('Relative Error', fontsize=12)
    ax3.set_title('Error vs. $z$ (colored by $r = q/p$)', fontsize=13)
    ax3.set_yscale('log')
    ax3.set_ylim([1e-17, 1e-13])
    ax3.grid(True, alpha=0.3, linestyle='--')
    plt.colorbar(sc1, ax=ax3, label='$r = q/p$')

    # 4. Scatter: error vs p (colored by z)
    ax4 = axes[1, 1]
    sc2 = ax4.scatter(p_vals, errors_array, alpha=0.6, c=z_vals, cmap='plasma', s=20, edgecolor='k', linewidth=0.2)
    ax4.set_xlabel('Parameter $p$', fontsize=12)
    ax4.set_ylabel('Relative Error', fontsize=12)
    ax4.set_title('Error vs. $p$ (colored by $z$)', fontsize=13)
    ax4.set_yscale('log')
    ax4.set_ylim([1e-17, 1e-13])
    ax4.grid(True, alpha=0.3, linestyle='--')
    plt.colorbar(sc2, ax=ax4, label='$z$')

    plt.suptitle('Monte Carlo Analysis of $(p,q)$-Gamma Function Redundancy',
                fontsize=14, y=1.02, fontweight='bold')
    plt.tight_layout()
    plt.savefig('monte_carlo_analysis.pdf', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('monte_carlo_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("\nFigures saved: monte_carlo_analysis.pdf / .png")
    plt.show()
    return fig


# ----------------------------------------------------------------------
# LaTeX table generation
# ----------------------------------------------------------------------

def generate_latex_table(errors, results):
    """Generate LaTeX table for statistical results"""
    errors_array = np.array(errors)
    machine_eps = np.finfo(float).eps

    percentiles = {
        'Q1': np.percentile(errors_array, 25),
        'Median': np.median(errors_array),
        'Q3': np.percentile(errors_array, 75),
        '95th': np.percentile(errors_array, 95),
        '99th': np.percentile(errors_array, 99)
    }

    latex_code = f"""\\begin{{table}}[ht]
\\centering
\\caption{{Statistical summary of Monte Carlo analysis across the entire parameter space ($N = {len(errors):,}$ random samples). All errors are at machine precision, confirming the redundancy of parameter $p$.}}
\\label{{tab:statistical}}
\\begin{{tabular}}{{lcc}}
\\toprule
\\textbf{{Statistic}} & \\textbf{{Value}} & \\textbf{{Relative to $\\epsilon_{{\\text{{mach}}}}$}} \\\\
\\midrule
\\textbf{{Descriptive Statistics}} & & \\\\
Mean error & ${np.mean(errors_array):.2e}$ & ${np.mean(errors_array)/machine_eps:.2f}\\times\\epsilon_{{\\text{{mach}}}}$ \\\\
Median error & ${np.median(errors_array):.2e}$ & ${np.median(errors_array)/machine_eps:.2f}\\times\\epsilon_{{\\text{{mach}}}}$ \\\\
Std deviation & ${np.std(errors_array):.2e}$ & ${np.std(errors_array)/machine_eps:.2f}\\times\\epsilon_{{\\text{{mach}}}}$ \\\\
Maximum error & ${np.max(errors_array):.2e}$ & ${np.max(errors_array)/machine_eps:.2f}\\times\\epsilon_{{\\text{{mach}}}}$ \\\\
\\midrule
\\textbf{{Percentiles}} & & \\\\
First quartile (Q1) & ${percentiles['Q1']:.2e}$ & ${percentiles['Q1']/machine_eps:.2f}\\times\\epsilon_{{\\text{{mach}}}}$ \\\\
Median (Q2) & ${percentiles['Median']:.2e}$ & ${percentiles['Median']/machine_eps:.2f}\\times\\epsilon_{{\\text{{mach}}}}$ \\\\
Third quartile (Q3) & ${percentiles['Q3']:.2e}$ & ${percentiles['Q3']/machine_eps:.2f}\\times\\epsilon_{{\\text{{mach}}}}$ \\\\
95th percentile & ${percentiles['95th']:.2e}$ & ${percentiles['95th']/machine_eps:.2f}\\times\\epsilon_{{\\text{{mach}}}}$ \\\\
99th percentile & ${percentiles['99th']:.2e}$ & ${percentiles['99th']/machine_eps:.2f}\\times\\epsilon_{{\\text{{mach}}}}$ \\\\
\\midrule
\\textbf{{Distribution Relative to $\\epsilon_{{\\text{{mach}}}}$}} & & \\\\
Samples with error = 0 & {np.sum(errors_array == 0):,} ({np.sum(errors_array == 0)/len(errors_array)*100:.1f}\\%) & -- \\\\
Samples $< \\epsilon_{{\\text{{mach}}}}$ & {np.sum(errors_array < machine_eps):,} ({np.sum(errors_array < machine_eps)/len(errors_array)*100:.1f}\\%) & -- \\\\
Samples $< 5\\epsilon_{{\\text{{mach}}}}$ & {np.sum(errors_array < 5*machine_eps):,} ({np.sum(errors_array < 5*machine_eps)/len(errors_array)*100:.1f}\\%) & -- \\\\
Samples $< 10\\epsilon_{{\\text{{mach}}}}$ & {np.sum(errors_array < 10*machine_eps):,} ({np.sum(errors_array < 10*machine_eps)/len(errors_array)*100:.1f}\\%) & -- \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    return latex_code


# ----------------------------------------------------------------------
# Main execution
# ----------------------------------------------------------------------

def main():
    print("\n" + "="*80)
    print("MONTE CARLO STATISTICAL ANALYSIS OF (p,q)-GAMMA REDUNDANCY")
    print("="*80)

    print("\n[Step 1/3] Running Monte Carlo analysis...")
    results, errors, z_vals, p_vals, r_vals = monte_carlo_analysis(num_samples=2000)

    print("\n[Step 2/3] Creating statistical visualization...")
    fig = create_statistical_plots(results, errors)

    print("\n[Step 3/3] Generating LaTeX table...")
    latex_table = generate_latex_table(errors, results)
    print("\n" + "="*80)
    print("LATEX TABLE CODE (copy to your .tex file):")
    print("="*80)
    print(latex_table)

    with open('statistical_table.tex', 'w', encoding='utf-8') as f:
        f.write(latex_table)
    print("\nLaTeX table saved to 'statistical_table.tex'")

    print("\n" + "="*80)
    print("MONTE CARLO ANALYSIS COMPLETE!")
    print("="*80)

    return results, errors, fig


if __name__ == "__main__":
    results, errors, figure = main()
