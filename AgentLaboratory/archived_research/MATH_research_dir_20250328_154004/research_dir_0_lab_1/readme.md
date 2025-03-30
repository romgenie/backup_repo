# Adaptive Graph-Guided Iterative Prompt Calibration (AGGIPC)

Welcome to the AGGIPC repository! This project explores an adaptive graph-guided iterative prompt calibration framework designed to improve mathematical reasoning in large language models (LLMs) by leveraging dynamic reasoning graphs and iterative self-correction mechanisms. Although our current experimental implementation on the MATH500 benchmark did not yield performance improvements over standard prompting, the approach provides valuable insights into refining prompt engineering strategies and error detection heuristics.

---

## Table of Contents

- [Overview](#overview)
- [Motivation and Contributions](#motivation-and-contributions)
- [Methodology](#methodology)
- [Experimental Setup](#experimental-setup)
- [Results](#results)
- [Discussion and Future Work](#discussion-and-future-work)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Citing This Work](#citing-this-work)
- [License](#license)

---

## Overview

The AGGIPC framework aims to enhance the chain-of-thought (CoT) reasoning of LLMs by:

- Constructing dynamic reasoning graphs in which each node represents an individual reasoning step.
- Quantifying reasoning quality using the error ratio:
  
  ε = n₍weak₎ / n₍total₎

  where n₍weak₎ is the number of flagged weak steps (using heuristic measures such as step length and semantic consistency) and n₍total₎ is the total number of reasoning steps.
- Applying iterative re-prompting to re-elaborate on steps identified as weak while trying to preserve the overall logical flow.
  
Despite the innovative design, experiments on a dataset of 500 math problems (MATH500) revealed that while the baseline system achieved 70.2% accuracy under standard prompting, the current AGGIPC implementation yielded an accuracy of 0.0% due to excessive re-prompting.

---

## Motivation and Contributions

The motivation behind AGGIPC is twofold:
- Enhance the adaptability of prompt-based reasoning in LLMs by locally refining weak reasoning segments.
- Balance correction efforts with overall inference cost via a constrained optimization formulation.

Key contributions include:
- Integration of dynamic reasoning graphs for localized error identification.
- Formulation of the error ratio (ε) and framing the correction process as a constrained optimization problem.
- Detailed experimentation illustrating the challenges of overcorrection in iterative self-correction systems.
- Insights and analysis that inform potential improvements in adaptive prompt engineering.

---

## Methodology

The core components of AGGIPC include:

1. **Dynamic Reasoning Graphs**  
   - Extract reasoning steps from the model's chain-of-thought output.
   - Construct a graph G = (V, E) where vertices V represent individual reasoning steps and edges E capture logical dependencies.

2. **Error Detection and Quality Ratio**  
   - Evaluate each reasoning step using heuristic measures (e.g., a simple length-based criterion where a step with fewer than 15 characters is flagged as weak).
   - Compute the error ratio:
     
     ε = (number of weak steps) / (total number of steps)

3. **Iterative Correction**  
   - Formulate a constrained optimization problem:
     
     minimize ε(S) subject to a coherence score C(S) ≥ γ  
     
     where S is the reasoning chain and γ is a pre-defined threshold.
   - Apply targeted re-prompting using a fixed template and a simple Process Reward Model (PRM) that rewards consistency in reasoning.

4. **Implementation Details**
   - Multi-threading is used via a ThreadPoolExecutor with a worker count determined by min(20, CPU cores) to balance processing time and system load.
   - One re-prompt iteration is applied per problem when weak steps are detected.

For more details, refer to the included LaTeX paper ([paper.tex](./paper.tex)) in this repository.

---

## Experimental Setup

Our experiments were conducted on the MATH500 benchmark consisting of 500 diverse mathematical problems with reference solutions. Key hyperparameters include:

- **Dataset Size**: 500 problems
- **Baseline Accuracy**: 70.2%
- **Re-Prompt Iterations**: 1 (if weak steps detected)
- **Weak Step Criterion**: Flag any reasoning step shorter than 15 characters
- **Error Ratio (ε)**: Computed as n₍weak₎/n₍total₎
- **ThreadPool Workers**: min(20, CPU cores)

The evaluation metric is the accuracy defined by the percentage of correct final answers.

---

## Results

The AGGIPC framework in its current implementation resulted in:
- **AGGIPC Accuracy**: 0.0% (0 out of 500 correct)
- **Baseline Accuracy**: 70.2%
- **Iterative Correction Rate**: Nearly 100% (almost all examples underwent re-prompting)

These results, along with the accompanying plots and tables provided in the paper, underscore the difficulty in balancing iterative corrections with the preservation of coherent reasoning flows.

---

## Discussion and Future Work

While the AGGIPC approach is conceptually innovative, the current extensive re-prompting mechanism (primarily based on a simplistic length-based weak step detection) leads to significant performance degradation. Key discussion points include:

- **Heuristic Limitations**:  
  The simple step-length heuristic results in misidentification of valid concise reasoning steps, causing unnecessary re-correction.
  
- **Graph-Guided Re-Prompting**:  
  Though dynamic reasoning graphs are a promising idea, the present implementation does not fully leverage the potential of a structured logical representation.
  
- **Future Directions**:
  - Refine error-detection criteria using semantic analysis or embedding-based similarity metrics.
  - Implement adaptive thresholds for re-prompting to avoid overcorrection.
  - Explore rollback mechanisms to revert to earlier coherent states if further corrections worsen the chain-of-thought.
  - Integrate additional validation layers (e.g., external mathematical solvers) for intermediate consistency checks.

These insights will guide future iterations aimed at developing a more robust self-correction mechanism.

---

## Repository Structure

The repository is organized as follows:

- **/src**: Source code for AGGIPC implementation, including the dynamic graph construction, heuristic error detection, and iterative re-prompting logic.
- **/experiments**: Scripts and notebooks replicating the experiments on the MATH500 dataset.
- **/figures**: Generated figures (bar charts, line plots) illustrating iterative correction incidence and cumulative accuracy.
- **paper.tex**: The LaTeX source file of the research paper detailing the methodology, experimental setup, and discussion.
- **README.md**: This readme file.

---

## Installation

To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/aggipc.git
cd aggipc
pip install -r requirements.txt
```

*Note: Please ensure Python 3.7 or higher is installed on your system.*

---

## Usage

The main experimental script can be executed as follows:

```bash
python src/run_experiment.py --dataset MATH500 --max_iterations 1
```

Additional command-line arguments are available to customize the error threshold, number of threads, and other hyperparameters. See the help message for more details:

```bash
python src/run_experiment.py --help
```

---

## Citing This Work

If you find this project useful for your research, please consider citing our paper:

```
@article{aggipc2023,
  title={Adaptive Graph-Guided Iterative Prompt Calibration for Math Reasoning},
  author={Agent Laboratory},
  year={2023},
  note={Preprint available on arXiv}
}
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

---

Thank you for your interest in AGGIPC. We welcome feedback, suggestions, and contributions to help improve the framework and explore more robust adaptive self-correction methods for mathematical reasoning in large language models.

Happy Coding!