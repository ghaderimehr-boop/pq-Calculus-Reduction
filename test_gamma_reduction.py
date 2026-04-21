"""
test_gamma_reduction.py – Quick validation of (p,q)-gamma reduction to q-gamma.
Tests the identity Gamma_{p,q}(z) = Gamma_{q/p}(z) numerically.
All core functions are imported from pq_core (exact original implementations).
"""

import math
from pq_core import q_gamma, pq_gamma

# ============================================================================
# Test 1: Direct equality Gamma_{p,q}(z) = Gamma_{q/p}(z)
# ============================================================================

def test_gamma_equality():
    """Test equality of gamma functions"""
    test_cases = [
        (1.5, 0.9, 0.6),
        (2.3, 0.8, 0.4),
        (0.7, 0.95, 0.7),
        (2.0, 0.85, 0.3),
        (3.1, 0.75, 0.5),
    ]
    
    print("Testing: Gamma_{p,q}(z) = Gamma_{q/p}(z)")
    print("=" * 60)
    
    for z, p, q in test_cases:
        gamma_pq = pq_gamma(z, p, q)
        r = q / p
        gamma_r = q_gamma(z, r)
        rel_error = abs(gamma_pq - gamma_r) / (abs(gamma_pq) + 1e-15)
        
        print(f"z={z:.2f}, p={p:.3f}, q={q:.3f}, r=q/p={r:.6f}")
        print(f"  Gamma_{{p,q}} = {gamma_pq:.12f}")
        print(f"  Gamma_r      = {gamma_r:.12f}")
        print(f"  Relative error = {rel_error:.2e}")
        
        if rel_error < 1e-10:
            print("  OK: Equality confirmed")
        else:
            print("  FAIL: Equality not satisfied")
        print()


# ============================================================================
# Test 2: Dependence on p with fixed ratio r
# ============================================================================

def test_p_dependence():
    """Check whether changing p affects the value when r = q/p is fixed"""
    z = 2.5
    r = 0.7  # fixed ratio
    
    print("Testing dependence on p with fixed r")
    print("=" * 60)
    print(f"z = {z}, r = {r}")
    print()
    
    p_values = [0.99, 0.9, 0.8, 0.7, 0.6]
    
    for p in p_values:
        q = r * p
        if q >= p or q <= 0 or p > 1:
            continue
            
        gamma_pq = pq_gamma(z, p, q)
        gamma_r = q_gamma(z, r)
        rel_error = abs(gamma_pq - gamma_r) / (abs(gamma_pq) + 1e-15)
        
        print(f"p={p:.3f}, q={q:.6f}, r={q/p:.6f}")
        print(f"  Gamma_{{p,q}} = {gamma_pq:.12f}")
        print(f"  Difference from Gamma_r = {abs(gamma_pq - gamma_r):.2e}")
        print(f"  Relative error = {rel_error:.2e}")
        
        if rel_error < 1e-10:
            print("  OK: p has no effect")
        else:
            print("  WARNING: p has an effect")
        print()


# ============================================================================
# Test 3: Systematic analysis over parameter space
# ============================================================================

def systematic_analysis():
    """Systematic analysis over parameter space"""
    z_values = [0.5, 1.2, 2.0, 3.5]
    r_values = [0.3, 0.5, 0.7, 0.9]
    
    print("Systematic analysis: effect of p with fixed r")
    print("=" * 60)
    
    results = []
    
    for z in z_values:
        for r in r_values:
            print(f"\nz = {z:.1f}, r = {r:.1f}")
            print("-" * 40)
            
            p_test = [0.99, 0.9, 0.7, 0.5]
            errors = []
            
            for p in p_test:
                q = r * p
                if q >= p or q <= 0 or p > 1:
                    continue
                    
                gamma_pq = pq_gamma(z, p, q)
                gamma_r = q_gamma(z, r)
                rel_error = abs(gamma_pq - gamma_r) / (abs(gamma_pq) + 1e-15)
                errors.append(rel_error)
                print(f"  p={p:.2f}, q={q:.4f}: error = {rel_error:.2e}")
            
            if errors:
                max_error = max(errors)
                mean_error = sum(errors) / len(errors)
                print(f"  Mean error: {mean_error:.2e}, Max error: {max_error:.2e}")
                
                if max_error < 1e-10:
                    print("  OK: p is completely redundant for this (z, r)")
                elif max_error < 1e-6:
                    print("  WARNING: p is almost redundant (small error)")
                else:
                    print("  FAIL: p has significant effect")
                
                results.append((z, r, mean_error, max_error))
    
    return results


# ============================================================================
# Test 4: Check for scaling factor independent of z
# ============================================================================

def check_functional_relation():
    """Check whether Gamma_{p,q}(z) = C * Gamma_r(z) with C independent of z"""
    print("\nChecking for scaling factor independent of z")
    print("=" * 60)
    
    z1, z2 = 1.5, 2.5
    p, q = 0.8, 0.4
    r = q / p
    
    gamma_pq_z1 = pq_gamma(z1, p, q)
    gamma_r_z1 = q_gamma(z1, r)
    gamma_pq_z2 = pq_gamma(z2, p, q)
    gamma_r_z2 = q_gamma(z2, r)
    
    C1 = gamma_pq_z1 / gamma_r_z1
    C2 = gamma_pq_z2 / gamma_r_z2
    
    print(f"p={p}, q={q}, r={r}")
    print(f"For z={z1}: Gamma_{{p,q}}/Gamma_r = {C1:.12f}")
    print(f"For z={z2}: Gamma_{{p,q}}/Gamma_r = {C2:.12f}")
    print(f"Relative difference: {abs(C1 - C2)/abs(C1):.2e}")
    
    if abs(C1 - C2) < 1e-12:
        print("OK: Scaling factor C is independent of z")
        if abs(C1 - 1.0) < 1e-12:
            print("  Moreover C = 1, so Gamma_{p,q} = Gamma_r")
        else:
            print(f"  But C != 1, yet independent of z")
    else:
        print("FAIL: Scaling factor depends on z")


# ============================================================================
# Test 5: Scaling effect analysis of parameter p
# ============================================================================

def analyze_scaling_effect():
    """Analyze whether p merely scales the independent variable"""
    print("\nAnalyzing scaling effect of p")
    print("=" * 60)
    
    z = 2.5
    r = 0.6  # fixed ratio
    
    p1, p2 = 0.9, 0.6
    q1 = r * p1
    q2 = r * p2
    
    gamma_p1q1 = pq_gamma(z, p1, q1)
    gamma_p2q2 = pq_gamma(z, p2, q2)
    gamma_r = q_gamma(z, r)
    
    print(f"z = {z}, r = {r}")
    print(f"p1={p1}, q1={q1}: Gamma = {gamma_p1q1:.12f}")
    print(f"p2={p2}, q2={q2}: Gamma = {gamma_p2q2:.12f}")
    print(f"Gamma_r = {gamma_r:.12f}")
    
    error1 = abs(gamma_p1q1 - gamma_r) / abs(gamma_r)
    error2 = abs(gamma_p2q2 - gamma_r) / abs(gamma_r)
    
    print(f"\nRelative error for p1: {error1:.2e}")
    print(f"Relative error for p2: {error2:.2e}")
    
    if error1 < 1e-10 and error2 < 1e-10:
        print("\nOK: p is only a scaling factor, adds no new information")
    else:
        print("\nFAIL: p has significant effect")


# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    print("Algebraic analysis of q-gamma and (p,q)-gamma functions")
    print("=" * 60)
    
    test_gamma_equality()
    test_p_dependence()
    results = systematic_analysis()
    check_functional_relation()
    analyze_scaling_effect()
    
    print("\n" + "=" * 60)
    print("Preliminary conclusion:")
    print("=" * 60)
    
    if results:
        all_errors = [max_err for (_, _, _, max_err) in results]
        max_overall_error = max(all_errors)
        
        if max_overall_error < 1e-12:
            print("OK: Strong evidence that Gamma_{p,q}(z) depends only on r = q/p")
            print("    Parameter p is informationally redundant")
        elif max_overall_error < 1e-8:
            print("WARNING: Good evidence that p is almost redundant (tiny numerical error)")
        elif max_overall_error < 1e-4:
            print("WARNING: Moderate evidence – p may have a small effect")
        else:
            print("FAIL: Weak evidence – p appears to have significant effect")
        
        print(f"\nMaximum error observed across all tests: {max_overall_error:.2e}")
