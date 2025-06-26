# DIgital Spectroscopy & COrrection of Circuit noise  (DISCO) 🚀  
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)  
> ⚠️ This publicly visible project accompanies our published paper: [Phys. Rev. A 111, 062605 (2025)](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.111.062605). Parts of the implementation are shared for preview and reference only.

## 🧠 What This Is About

Achieving fault-tolerant quantum computing means fighting noise — and doing it *smartly*, with insight into how noise actually interacts with control. This project introduces a **resource-efficient** strategy to learn and mitigate noise in digital quantum circuits, especially under *quantumly correlated*, *non-Gaussian*, and *non-Markovian* conditions — in other words, where the system dynamics display *high irreducible complexity*.

We flip the conventional approach:  
❌ Don’t learn *all* possible noise.  
✅ Learn only what matters to your control.

This leads to a powerful two-step workflow:
1. **Learning noise** via _control-adapted quantum noise spectroscopy_ (QNS) — to extract only the parts of the spectrum that impact your actual digital control.
2. **Optimizing control** based on the learned spectra — to design noise-tailored quantum circuits that actively suppress or avoid those noise contributions.

Here's a high-level schematic:

![Highlight Diagram](https://i.imgur.com/UszQlQI.png)

---

## 🌟 Key Features

- ✅ **Control-Adapted Noise Spectroscopy (CA-QNS)**  
  Learn only what matters — extract reduced spectra directly tied to the control you're using. A powerful concept introduced by Gerardo Paz-Silva.

- ✅ **Fundamental Digital QNS Protocols**  
  Tackle strong, non-Gaussian noise using a _bounded_ number of control sequences — avoiding the curse of dimensionality, even under ultra-strong coupling.

- ✅ **Noise-Tailored Circuit Optimization**  
  Use learned spectra to shape digital pulse sequences that mitigate noise *in practice*.

- ✅ **Built-In Symmetry Analysis**  
  Reduce learnable spectra using _binding symmetry_ and _dark-spectrum filtering_ — key for scalable estimation.

- ✅ **Beyond the Single-Qubit Regime**  
  We demonstrate scalability to two-qubit systems (and beyond), using rigorous Monte Carlo simulations and digital-circuit “surgery” under noise — a rare feat for control-based QNS methods.

---

## 📂 Project Structure

- `one-qubit Demo/` — Demonstration of single-qubit DISCO using fundamental digital QNS.  
- `two-qubit Small Circuit` — Demonstration of small circuit (two-qubit, circuit layer-2) DISCO using fundamental digital QNS.  


---

## 🧪 Who Might Find This Useful?

- ⚛️ Quantum researchers working on **control**, **noise spectroscopy**, or **hardware-aware benchmarking**
- 🧑‍💻 Engineers in **nonlinear/stochastic systems** and **data-driven noise learning**
- 👩‍🔬 Experimental physicists exploring **non-Gaussian environments** in superconducting or spin qubit platforms

---

## 🔗 Connect with Me

[![Google Scholar](https://img.shields.io/badge/Scholar-Profile-blue)](https://scholar.google.com/citations?user=XXXX)  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/XXXX)  
📧 [wenzheng.dong.quantum@gmail.com](mailto:wenzheng.dong.quantum@gmail.com)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](./LICENSE) file for details.
