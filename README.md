# Steiner Tree Cutting Planes

Automating cardinality-based cutting planes for the Steiner Tree Problem, and an independent lift-and-project method - a research internship project.

This project reproduces and automates a set of cutting planes for the Steiner Tree Problem on the **EMCF-2** formulation, and adds an independent lift-and-project method. The formulation and the three cardinality cuts are from my professor's research; my contribution is the automation, the validation, and the write-up.

> **Note on the formulation and cuts.** The EMCF-2 formulation and the three cardinality cut families (the k-arc cardinality cutset inequality, Extended Cut II, and the cardinality matching inequality) are from the research of my internship supervisor **Prof. Md Shahrukh Anjum** at **Ahmedabad University**. This repository is my implementation and automation of that work. The lift-and-project method is my own.

---

## What this is

The Steiner Tree Problem: given a graph with edge costs and a set of *terminal* nodes, find the cheapest set of edges that connects all the terminals (using other nodes as helpers if that is cheaper). It is a classic NP-hard problem.

My professor's paper builds an extended formulation (EMCF-2) and three families of cutting planes that tighten its lower bound. In the paper, the cutsets those cuts are built on are chosen **by hand**. The goal of my internship was to **automate that**, find the cutsets automatically and add the cuts round by round and to see how much of the root gap could be closed.

The repository contains two methods:

1. **His cardinality cuts, automated** (`emcf2_with_cuts.py`) - EMCF-2 with the three cuts, added automatically. Because some automatically found cutsets do not satisfy the assumption the cuts need, validity is **verified against the known optimum** (a cut is kept only if the bound stays at or below it). The optimum is used only as a check, never to build a cut.
2. **Lift-and-project, my own** (`lp_cglp_gurobi.py`) - cuts on the bidirected model that are valid by construction, so they need **no knowledge of the optimum** and can never push the bound past the true answer.

The full story - where I started, the approach, the dead ends, and how I worked through them is in `Project_Report.pdf`.

---

## Data

All experiments use the **SteinLib** benchmark library specifically the `I080` and `I160` instance sets. SteinLib is a standard public collection of Steiner tree test instances.

- SteinLib: http://steinlib.zib.de/

The instance files are `.stp` format. (SteinLib instances are not redistributed here; download them from the link above.)

---

## Results (short version)

- **Lift-and-project** (needs no optimum): closes the root LP–IP gap fully on **13 of 15** instances, valid by construction. The two that do not fully close are i080-214 and i080-344.
- **Cardinality cuts** (verified against known optima): close i080-235, i080-305, i080-331 fully and i080-332 to ~99%, and do nothing on instances with no exploitable cutset.
- **Comparison with the solver:** modern Gurobi solves every one of these instances to optimality in **seconds** (a cut loop here can take minutes to a couple of hours). The value of the project is in understanding the formulation and the bound, not in competing with the solver, this is discussed in the report.

---

## Honest notes

- The cardinality-cut results are **verified against the published SteinLib optima**; they are not produced with no knowledge of the answer. The lift-and-project results are the ones that need no optimum.
- The hard part turned out to be **selecting the right cutsets**, the step the professor does by eye. Several automatic ways to do this without the optimum were tried and did not work reliably; this is written up honestly in the report.

---

## Files

- `emcf2_with_cuts.py` - EMCF-2 with the three cardinality cuts, automated (run with the known optimum).
- `lp_cglp_gurobi.py` - the independent lift-and-project cuts (no optimum needed).
- `core.py` - shared helper (reads the `.stp` file, builds the graph). Needed by both scripts.
- `HOW_TO_RUN.pdf` - setup and run instructions.
- `Project_Report.pdf` - the full project write-up.

## Running it

See `HOW_TO_RUN.pdf`. In short (with SteinLib `.stp` files and a Gurobi academic licence):

```bash
python emcf2_with_cuts.py i080-305.stp 5932      # cardinality cuts, verified against the optimum
python lp_cglp_gurobi.py i080-305.stp --ip       # lift-and-project, no optimum needed
```

---

## Acknowledgements

Done as a research internship under my supervisor **Prof. Md Shahrukh Anjum** at **Ahmedabad University**, whose EMCF-2 formulation and cardinality cuts this work implements and automates. Benchmark instances are from the SteinLib library.

## Disclaimer

This is a research/educational project. The formulation and cardinality cuts are the intellectual work of my supervisor and his co-authors; this repository is an implementation shared with permission.
