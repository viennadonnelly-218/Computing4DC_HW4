# Diabetes Mellitus Prediction – Computing for Data Science HW4

## Group members
- Jose Luis Fernández
- Vienna Donnelly
- Mariana Troncoso

## Project structure
- `src/computing4dc_hw4/hw4.py`: exercises 1–4 (functional programming and recursion)
- `src/diabetes_analysis.ipynb`: imports the library and runs all the steps
- `src/diabetes_library/importing/loading.py`: load the data and split it into train/test (steps a, b)
- `src/diabetes_library/preprocessing/cleaning.py`: drop and fill missing values (steps c, d)
- `src/diabetes_library/preprocessing/encoding.py`: one-hot and binary encoding (steps e, f)
- `src/diabetes_library/modeling/training.py`: train the model and add predictions (steps g, h)
- `src/diabetes_library/modeling/evaluation.py`: compute the ROC AUC (step i)
- 

## How to run
1. Install the environment: `uv sync`
2. Open `diabetes_analysis.ipynb` with the project's `.venv` kernel and run all cells.