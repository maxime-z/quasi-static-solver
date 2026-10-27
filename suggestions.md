# Suggestions on `doc.md` (do not merge blindly)

This file is a review of `quasi-static-solver/doc.md`. It does **not** replace that document. Use both together when writing the implementation plan.

The original doc is a Cursor-facing roadmap: a from-scratch 3D mixed FEM Darwin solver for 2.05 MHz low-field MRI, with GPU matrix-free operators and implicit-function automatic differentiation, covering an RF frequency-domain solver and a later gradient time-domain MQS solver.

## 1. What the document is trying to do

Three goals are stacked in one roadmap:

1. **Learn the numerics.** Hand-write Nédélec + Lagrange mixed FEM, gauge, and a custom solver.
2. **Ship a 2.05 MHz RF field solver** that keeps R, L, and C (Darwin) so coil tuning is not an afterthought.
3. **Make the solve differentiable** so coil / shield geometry can later be optimized.

Those are compatible, but they are not the same project. A from-scratch GPU AD solver is a multi-year stack. A Darwin RF solver that can be trusted on a coil+load+shield mesh is already a large Phase 1. The later plan should sequence *correctness of the mixed form* far ahead of CUDA and AD.

## 2. What is already solid

- Low-frequency breakdown of full-wave Maxwell at 2.05 MHz is a real reason not to start from a Helmholtz solver. Free-space wavelength is ~146 m; a 10–20 cm coil is electrically small.
- Pure MQS drops displacement current; pure EQS drops induction. A tuned MRI coil needs both, so Darwin is the right intermediate model *in principle*.
- Nédélec (first family, lowest order) for \(\boldsymbol{A}\) and Lagrange for \(\phi\) is the standard stable pair.
- Implicit-function / adjoint reverse mode for a linear solve is the right AD strategy. Unrolling Krylov in autograd is not.
- CPU validation before GPU, GPU before AD, time-domain last: that order is right.

## 3. Relation to the existing `solenoidal/` code

`doc.md` reads as if the solver starts from zero. The repo already has a working 2.05 MHz design loop:

| Piece | What it already does |
| --- | --- |
| `fem_mqs.py` | Axisymmetric MQS, unknown \(u = r A_\varphi\), energy \(L\), flux \(L\), sample/shield loss, \(B_1\) |
| `fem_c.py` | Axisymmetric electrostatic \(C_\text{eq}\) to the bore shield |
| `circuit.py` / `case_155mm.py` | Segment capacitors, series \(C_t\), match, \(Q\), S11 |
| `results/validation.json` | Filament vs FEM, capacitance, **displacement-current audit** |

The displacement audit is the most important existing fact:

- \(\Delta R_\text{sample} / R \approx 1.2\times 10^{-4}\)
- \(\Delta L / L \approx 2.4\times 10^{-5}\)
- The current code already decided `keep_MQS: true` for tissue at 2.05 MHz.

So Darwin is **not** justified by tissue \(\varepsilon\) at this frequency. \(\omega\varepsilon/\sigma\) in a 0.6 S/m load is a few percent at most, and the FEM audit says the effect on coil \(L\) and sample \(R\) is negligible.

Darwin *is* justified if the 3D solver is meant to replace the **split** `MQS + electrostatic C + lumped circuit` pipeline with a single self-consistent field model, especially for:

- non-axisymmetric coils (birdcage, saddle, figure-8, 3D winding paths);
- distributed winding / shield capacitance that should not be a separate EQS solve;
- lumped capacitors and ports inside the same 3D mesh;
- later geometry AD.

The implementation plan should state that split-physics remains the production path for axisymmetric solenoids until the 3D Darwin code reproduces `solenoidal` numbers on a comparable case.

## 4. Physics and discrete form: fill the gaps

Section 2 is the right idea written too schematically to implement.

### 4.1 Write the Darwin system before any code

The slogan \(\varepsilon \partial^2\boldsymbol{A}/\partial t^2 \approx 0\) only drops the radiative \(-\omega^2\varepsilon\boldsymbol{A}\) term. The frequency-domain system that should go into the plan is along these lines (Coulomb-gauged Darwin):

\[
\nabla\times(\nu\nabla\times\boldsymbol{A})
+(\sigma+j\omega\varepsilon)(j\omega\boldsymbol{A}+\nabla\phi)
-j\omega\varepsilon\, j\omega\boldsymbol{A}
=\boldsymbol{J}_s
\]

which is equivalent to keeping conduction current plus the irrotational displacement current, and dropping the solenoidal displacement current that radiates. A cited weak form (Badics, Costabel/Zhao, or an equivalent Darwin-in-bounded-domain paper) should be copied into the solver notes. Do not invent the \(2\times 2\) blocks from the slogan.

Charge conservation / the \(\phi\) equation is not independent of Ampere. \(\boldsymbol{J}_s\) and \(\rho_s\) cannot be chosen independently; they must satisfy continuity. The RHS in `doc.md` is therefore misleading as a product interface. Drive the coil by **current-driven stranded windings and/or lumped ports**, and let \(\rho_s\) follow.

### 4.2 The matrix in §2.2 is incomplete

The printed block

\[
\begin{bmatrix} K_{AA} & K_{A\phi} \\ K_{\phi A} & K_{\phi\phi} \end{bmatrix}
\begin{bmatrix} \boldsymbol{A} \\ \phi \end{bmatrix}
=
\begin{bmatrix} \boldsymbol{J}_s \\ \rho_s \end{bmatrix}
\]

is missing:

- \(\omega\), \(\sigma(\mathbf{x})\), \(\varepsilon(\mathbf{x})\) (the blocks are complex and frequency-dependent);
- the gauge unknown \(p\) from §2.3, which makes a **3-field** saddle point \((\boldsymbol{A},\phi,p)\);
- how Dirichlet / PEC / stranded-current constraints are applied;
- lumped circuit rows (the actual reason Darwin was chosen).

If Lagrange-multiplier Coulomb gauge is kept, the discrete operator looks more like

\[
\begin{bmatrix}
K_{AA}(\omega) & K_{A\phi}(\omega) & G \\
K_{\phi A}(\omega) & K_{\phi\phi}(\omega) & 0 \\
G^\top & 0 & 0
\end{bmatrix}
\begin{bmatrix} A \\ \phi \\ p \end{bmatrix}
=
\begin{bmatrix} J \\ 0 \\ 0 \end{bmatrix}.
\]

A natural gauge for inhomogeneous \(\varepsilon\) is \(\nabla\cdot(\varepsilon\boldsymbol{A})=0\), not \(\nabla\cdot\boldsymbol{A}=0\).

**Suggestion for Phase 1:** do not start with an extra unknown \(p\). Prefer, in order:

1. compatible de Rham pair + tree–cotree (or spanning-tree) gauge on edges;
2. or a weakly imposed Coulomb / grad-div term;
3. Lagrange multiplier \(p\) only if (1)–(2) fail on the first analytic tests.

The extra saddle field is a solver problem, not a physics requirement for the first MVA.

### 4.3 Lumped capacitors and ports are the missing product feature

`doc.md` says MQS/EQS cannot handle L–C resonance, then never says how \(C_t\), segment capacitors, or a 50 Ω port enter the FEM. Without that, a 3D Darwin solve of coil + tissue still does not give SRF, \(Q\), or \(S_{11}\).

Pick one coupling model in the plan (recommended: field–circuit / modified nodal analysis):

- FEM edge DoFs on a port or capacitor terminal pair carry a voltage;
- a small MNA stamp for \(C\), \(R\), and the source sits in the same sparse matrix;
- post-process \(Z(\omega)\) and search the zero-reactance frequency.

Meshing a commercial capacitor body to get its own SRF is a different (and harder) problem. Do not list “lumped-capacitor SRF” as an early RF-solver deliverable unless the capacitor geometry is explicitly meshed.

### 4.4 Do not volume-mesh copper at 2.05 MHz

Copper skin depth at 2.05 MHz is tens of micrometres. A 45-turn 3D volume mesh of solid copper is not a Phase 1 (or Phase 2) mesh. Carry over the `solenoidal` practice:

- **stranded** windings: uniform \(J\) on a wire path or a coarse turn volume, copper \(R\) from an analytical / proximity model;
- **surface impedance** on the bore shield (already in `fem_mqs.py`);
- optional later: massive-conductor mode on a *short* coil, as in the existing 4-turn skin-effect check.

## 5. Verification: the Phase 1 test in `doc.md` is not a Darwin test

“Uniform magnetic field, conducting sphere eddy currents, error \(<0.5\%\)” is a classic **MQS** benchmark. It never exercises \(\phi\), \(\varepsilon\), gauge coupling, or lumped \(C\). A pure \(\boldsymbol{A}\)-only eddy-current code can pass it.

Replace that single criterion with a ladder. Suggested order:

| # | Test | What it proves | Pass band (first-order tets) |
| --- | --- | --- | --- |
| 1 | Manufactured solution or constant \(\boldsymbol{B}\) in a box, \(\sigma=0\) | Nédélec curl, edge orientation, PEC | energy / \(\boldsymbol{B}\) within a few % and converging |
| 2 | Conducting sphere in uniform \(\boldsymbol{B}\) (Smythe) | MQS eddy currents, \(\boldsymbol{A}\) only | 2–5 % on induced power at first mesh; finer later |
| 3 | Dielectric sphere in uniform \(\boldsymbol{E}\) | Lagrange \(\phi\), \(\varepsilon\) jumps | same |
| 4 | Two conductors: electrostatic \(C\) vs analytic / `fem_c` idea | \(K_{\phi\phi}\) | 5–10 % on coarse mesh |
| 5 | Single-loop + lumped \(C\): resonant \(f\) vs \(1/(2\pi\sqrt{LC})\) | Darwin + circuit coupling | ~1 % on \(f_0\) once \(L,C\) are each converged |
| 6 | Axisymmetric solenoid vs `solenoidal` MQS \(L\), \(B_1\), \(R_\text{sample}\) | regression against this repo | few % on \(L\) and \(B_1\) |
| 7 | Same solenoid + FEM \(C\) + lumped \(C_t\): resonance vs `case_155mm.py` | why Darwin exists | \(f_0\) within a stated kHz band |

Keep \(<0.5\%\) only as a *refined-mesh* target on tests 2–3, not as the MVA gate.

Also fix the name: the doc says MVA; the usual term is MVP (minimum viable product / 最小可行性). Harmless, but confusing in a Cursor prompt.

## 6. Linear algebra: the largest technical hole

The discrete Darwin operator is a **complex, non-Hermitian saddle point**. It is not SPD.

Consequences:

- CG is the wrong default. Use GMRES or BiCGStab (or COCG if the operator is complex-symmetric and that is proven).
- Unpreconditioned Krylov will stall as soon as the mesh is 3D and \(\sigma/\varepsilon\) contrast is MRI-like.
- The missing topic is **block preconditioning**: AMG (or Cholesky) on the Poisson-like \(\phi\) block; auxiliary-space Maxwell / AMS / HX or a cheap Jacobi/SSOR on the curl–curl block; Schur complement for the gauge if \(p\) is present.

Phase 1 should use `scipy.sparse` + a **direct** solver (`superlu` / `umfpack` / `sksparse.cholmod` on a real equivalent, or a complex sparse direct factor). Iterative + preconditioner is Phase 1b on CPU. GPU Krylov without a preconditioner is not a solver.

## 7. Architecture suggestions (keep the spirit, relax the bans)

### 7.1 Do not ban CSR in Phase 1

§4.2 “严禁 CPU 端组装 CSR” fights §5 Phase 1 (“`scipy.sparse`”). Sparse assembly is the correct way to prove the form. Rephrase the ban as a Phase 2 *goal*: do not *ship* a CPU-assembled CSR path as the large-mesh production path.

### 7.2 Pick one AD stack

“PyTorch (或 JAX)” will fork the code. Recommend **PyTorch** only, because the rest of the doc is already written around `torch.autograd.Function`. JAX’s `custom_linear_solve` is nicer for implicit AD, but switching later is cheaper than supporting both.

### 7.3 Do not write CUDA in Phase 2 until batched PyTorch is the bottleneck

A lowest-order tet Nédélec local matrix is \(6\times 6\). The cost is gather/scatter of edge DoFs, not flops. Suggested GPU ladder:

1. Same sparse matrix as Phase 1, factored or iterated on CPU.
2. Element matrices batched with `torch.einsum` (or numpy) — still assemble CSR, or do matrix-free matvec in Python.
3. Matrix-free matvec in Triton **if** (2) is slow and a preconditioner already exists.
4. Custom C++/CUDA extension last.

“One CUDA thread block per tet” is often occupancy-poor. Do not lock that kernel shape into the plan.

### 7.4 AD: start from materials, not moving nodes

Phase 3’s first test (move coil control nodes, loss = \(B_1\) homogeneity) is the hardest AD problem: Piola-mapped Nédélec, mesh quality, and shape derivatives. Suggested AD ladder:

1. Implicit-function adjoint around the **CPU sparse** solve (no GPU).
2. Finite-difference check on \(\partial J/\partial\sigma\), \(\partial J/\partial\varepsilon\) on a tiny mesh.
3. Current or stranded-turn amplitude as the design variable.
4. Only then node coordinates / CAD parameters.

Treat real and imaginary parts as \(\mathbb{R}^2\). Do not rely on complex autograd being correct by default.

Inverse design of a coil is often a circuit-parameter or current-density problem. Free-form mesh motion is not the first useful differentiable API.

## 8. Scope: RF first, gradient later

“一体两翼” is a good product picture and a bad first sprint.

- RF Darwin: dielectrics, lumped ports, complex frequency-domain, stranded coils.
- Gradient MQS: kHz pulses, copper shields, skin effect, time stepping, possibly thin-layer models.

Skin-depth and time-stepping requirements will wreck an RF mesh if they are designed together. Keep Phase 4 as a *later degeneration* of the MQS limit (`σ ≫ ωε`, drop \(\phi\) or freeze it), not as a co-equal deliverable.

## 9. Software shape the later plan should assume

Suggested package under `quasi-static-solver/` (names can change):

```text
quasi-static-solver/
  doc.md
  suggestions.md
  darwin/
    mesh.py          # meshio / gmsh tet mesh, materials, edge/face maps
    nedelec.py       # lowest-order Nédélec on tets, Piola, edge signs
    lagrange.py      # P1 nodes
    forms.py         # element K_AA, K_Aφ, K_φφ, gauge
    assemble.py      # scipy.sparse CSR (Phase 1)
    gauge.py         # tree-cotree or LM
    circuit.py       # lumped C/R/port stamps
    solve.py         # sparse direct, then iterative
    post.py          # B = curl A, E = -jωA - ∇φ, energy, SAR, Z(ω)
    validate/        # tests 1–7 above
```

Stay out of GPU and `torch.autograd.Function` until `validate/` passes on CPU.

Reuse `solenoidal` as an oracle, not as a library to rewrite. Axisymmetric NGSolve H(curl) can optionally be a **Phase 0** physics freeze: implement the Darwin weak form in NGSolve, match tests 2–5, *then* reimplement shape functions from scratch and match NGSolve DoF by DoF. That avoids debugging the PDE and the Nédélec basis at the same time. If “from scratch” is non-negotiable, still keep NGSolve as a reference solution, not as the production engine.

## 10. Small corrections in `doc.md` (leave the file as-is)

- “Nédon 单第一类棱边有限元” → Nédélec first family edge elements.
- Title line “GPU 加行” → 加速.
- \(\boldsymbol{\phi}\) in the unknown vector should be scalar \(\phi\); \(\boldsymbol{\rho}_s\) is fine as a density.
- “集成电路/集总参数电容的自谐振频率” mixes IC-capacitor SRF with coil-mesh resonance. Split those deliverables.
- Phase 1 “MVA” → MVP.
- CG / BiCGStab listed as interchangeable; they are not (see §6).
- `text{grad_output}` in the adjoint equation should be written as a real residual of the loss with respect to the field (Wirtinger / stacked real-imag). The adjoint operator is \(A(p)^\top\) only in the real stacked form; for complex FEM the adjoint is conjugate-transpose of the *real* 2×2 block layout, not casually \(A^T\).

## 11. Recommended stance for the implementation plan

Keep Darwin, mixed FEM, CPU-first, GPU later, AD last.

Change these defaults when generating the plan:

1. Phase 1 is a **CPU sparse Darwin + lumped-circuit** MVP with a verification ladder, not a conducting-sphere MQS demo.
2. Gauge starts as tree–cotree (or NGSolve reference), not Lagrange multiplier \(p\).
3. Coils are stranded; shields use surface impedance; no copper skin mesh.
4. CSR assembly is allowed until the form is trusted.
5. Iterative GPU Krylov is blocked on having a CPU preconditioner that actually converges.
6. AD starts from material / current parameters on the CPU solver.
7. `solenoidal/` is the regression oracle for the axisymmetric coil.

If those are accepted, the first generated plan should be Phase 1 only, cut into files in `darwin/`, each gated by a numbered test from §5.
