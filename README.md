# Diabetes Mellitus Prediction – Computing for Data Science HW4

## Group members
- Jose Luis Fernández
- Vienna Donnelly
- Mariana Troncoso

## Project structure
- `src/computing4dc_hw4/hw4.py`: exercises 1–4 (functional programming and recursion)
- `src/computing4dc_hw4/data/loading.py`: load the data and split it into train/test (steps a, b)
- `src/computing4dc_hw4/preprocessing/cleaning.py`: drop and fill missing values (steps c, d)
- `src/computing4dc_hw4/preprocessing/encoding.py`: one-hot and binary encoding (steps e, f)
- `src/computing4dc_hw4/modeling/training.py`: train the model and add predictions (steps g, h)
- `src/computing4dc_hw4/modeling/evaluation.py`: compute the ROC AUC (step i)
- `notebooks/diabetes_analysis.ipynb`: imports the library and runs all the steps

## How to run
1. Install the environment: `uv sync`
2. Open `notebooks/diabetes_analysis.ipynb` with the project's `.venv` kernel and run all cells.