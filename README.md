# Project Overview

This project contains Python code to run the code related to the article Regulation of whole-body and segmental angular momentum during sidestep cutting [IN SUBMISSION].  
The main script (`main.py`) serves as the entry point for running the workflow, including any data processing, simulations, or analyses needed to reproduce the results.

---

# Requirements

- Python 3 (recommended: Python 3.9 or higher)
- `pip` (Python package manager)
- Git (optional, for cloning the repository)

All required Python packages are listed in `requirements.txt`.

---

# Setup Instructions

Follow these steps to set up the project locally.

## 1. Clone the repository

```bash
git clone <your-repo-url>
cd <your-repo-folder>
````

## 2. Create a virtual environment

Using Python 3, create a virtual environment:

```bash
python3 -m venv venv
```

## 3. Activate the virtual environment

* On macOS / Linux:

```bash
source venv/bin/activate
```

* On Windows:

```bash
venv\Scripts\activate
```

## 4. Install dependencies

Install all required packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

***

# Running the Code

To execute the project and generate results, run:

```bash
python3 main.py
```

This will run the full pipeline defined in the script.

***

# Output

* Results will typically be printed to the console and/or saved to files (depending on how `main.py` is implemented).
* Check the project directory for generated outputs such as logs, data files, or result summaries.


## Data Source and Citation

The original data used in this project comes from the following scientific article:

- https://pubmed.ncbi.nlm.nih.gov/40576072/

The corresponding dataset can be accessed and downloaded here:

- https://dataverse.nl/dataset.xhtml?persistentId=doi:10.34894/LZPY3B 【1-4f17f9】

If you use this repository or the provided data in your work, please consider citing both the original article and the dataset.

### Dataset citation (BibTeX)

```bibtex
@dataset{oonk2025markerless_dataset,
  author    = {Oonk, G. A. and Kempe, M. and Lemmink, K. A. P. M. and Buurke, T. J. W.},
  title     = {Validation of Markerless MoCap in dyadic team sports tasks},
  year      = {2025},
  publisher = {DataverseNL},
  doi       = {10.34894/LZPY3B},
  url       = {https://doi.org/10.34894/LZPY3B}
}
```

### Article citation (BibTeX)

```bibtex
@article{oonk2025markerless_article,
  author  = {Oonk, G. A. and Kempe, M. and Lemmink, K. A. P. M. and Buurke, T. J. W.},
  title   = {Examining the concurrent validity of markerless motion capture in dual-athlete team sports movements},
  journal = {Journal of Sports Sciences},
  year    = {2025},
  doi     = {10.1080/02640414.2025.2497678}
}
```

If this project or dataset has been useful for your research, please consider citing the above references to support the original authors and their work.




