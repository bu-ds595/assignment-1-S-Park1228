[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/jjj6D7QL)
# Assignment 1: Sampler Synthesis

**⏰ Due: Wednesday, February 18** | 15% of final grade

Design, implement, and critically analyze a novel MCMC sampling method.

📄 **[Assignment description (PDF)](assignment-01.pdf)** | 🤖 **[AI coding assistants guide](https://github.com/bu-ds595/course-materials-spring26/blob/main/notes/xx-agents.md)**

## Overview

In this assignment, you'll work with AI coding assistants to create a new MCMC sampler and rigorously evaluate it against standard baselines (Random Walk MH and HMC) on two challenging benchmark distributions.

## Structure

```
├── assignment-01.pdf             # Assignment description
├── assignment-01-starter.ipynb   # Starter code with baselines and benchmarks
├── requirements.txt              # Python dependencies
└── report/
    ├── report.tex                # Your report (NeurIPS format)
    └── neurips_2025.sty          # LaTeX style file
```

## Getting Started

### 1. Accept the assignment

Click the GitHub Classroom link shared by the instructor. This creates your own private copy of this repository.

### 2. Clone your repository

```bash
git clone https://github.com/bu-ds595/assignment01-YOUR_USERNAME.git
cd assignment01-YOUR_USERNAME
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Open the notebook

**VS Code:** Open the folder, then open `assignment-01-starter.ipynb`

**JupyterLab:** Run `jupyter lab` and open the notebook

**Google Colab:** Upload the notebook and run:
```python
!pip install jax jaxlib blackjax arviz
```

## Deliverables

1. **Code/Notebook** — Your sampler implementation with experiments and visualizations
2. **Written Report** — Maximum 3 pages in NeurIPS format (use `report/report.tex`)

## Submitting

```bash
git add .
git commit -m "Complete assignment 1"
git push
```

You can push multiple times. Only the final version at the deadline will be graded.

## Resources

- [Assignment description (PDF)](assignment-01.pdf)
- [AI coding assistants guide](https://github.com/bu-ds595/course-materials-spring26/blob/main/notes/xx-agents.md)
- [Course notes: MCMC & Variational Inference](https://bu-ds595.github.io/course-materials-spring26/notes/02-mcmc-vi.pdf)
- [BlackJAX documentation](https://blackjax-devs.github.io/blackjax/)
- [ArviZ documentation](https://python.arviz.org/)

