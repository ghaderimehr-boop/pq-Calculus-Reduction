[README.md](https://github.com/user-attachments/files/26948580/README.md)
# (p,q)-Calculus Reduction Project

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19687269.svg)](https://doi.org/10.5281/zenodo.19687269)

This repository contains the **first complete implementation** of the (p,q)-gamma function, (p,q)-Mittag-Leffler function, and associated fractional operators. It provides numerical verification for the theoretical result that the two-parameter (p,q)-calculus is **isomorphic** to the classical one-parameter q-calculus with effective parameter \( r = q/p \). The parameter \( p \) is proven to be redundant.

## 📄 Associated Paper

This code accompanies the paper:

> **"Occam’s Razor in Quantum Calculus: A Complete Reduction of (p,q)-Calculus to q-Calculus"**  
> *Authors: Mehran Ghaderi, Shahram Rezapour*  
> *Journal: Nature Communications (under review)*

## 🚀 Features

- **Stable implementation** of \(\Gamma_{p,q}(z)\) for real \(z>0\) and \(0<q<p\le 1\)
- **Stable implementation** of \(E_{\alpha,\beta}^{p,q}(z)\) with overflow protection
- **Monte Carlo statistical analysis** across the full parameter space (2000+ samples)
- **Sensitivity analysis** showing that \(\Gamma_{p,q}(z)\) depends only on \(r = q/p\)
- **Fractional relaxation equation solver** demonstrating identical dynamics
- **Publication‑ready figures** (PDF/PNG) generated automatically
- **All code is modular** – core functions are in `pq_core.py`, analysis scripts import them

## 📁 Repository Structure
