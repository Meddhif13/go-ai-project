# Project Setup Guide for Collaborators (Linux + VS Code)

## Prerequisites
Ensure you have the following installed on your system:
- **Git**
- **Git LFS** (Large File Storage)
- **Python 3.x**
- **Virtual environment (venv)**
- **Visual Studio Code (VS Code)** with the following extensions:
  - Python extension
  - GitLens (optional)

### Install Prerequisites
Open a terminal and run:
```bash
sudo apt update
sudo apt install git git-lfs python3 python3-venv -y
git lfs install
```

---

## Step 1: Clone the Repository
Open **VS Code** and in the terminal, run:
```bash
git clone https://github.com/Meddhif13/go-ai-project.git
cd go-ai-project
git lfs pull
```

---

## Step 2: Set Up Python Virtual Environment

1. Create a virtual environment:
   ```bash
   python3 -m venv venv
   ```

2. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### VS Code Python Interpreter Setup
- Open the Command Palette (`Ctrl + Shift + P`).
- Type `Python: Select Interpreter` and choose the `./venv` interpreter.

---

## Step 3: Configure the Dataset

1. Ensure the dataset file is available:
   - Download the dataset and place it in the `data/` folder:
     ```
     data/games.1000000.data
     ```

2. Confirm the dataset path in `config.py`:
   ```python
   DATASET_PATH = 'data/games.1000000.data'
   ```

---

## Step 4: Run the Project
With everything set up, run the main script:
```bash
python scripts/main.py
```

### Running the Script in VS Code
- Open `scripts/main.py` in VS Code.
- Press `F5` to run with the debugger.

---

## Troubleshooting

### Common Issues

1. **Git LFS files are not downloading**:
   ```bash
   git lfs install --force
   git lfs pull
   ```

2. **Dependencies not installing**:
   ```bash
   pip install --upgrade pip setuptools
   pip install -r requirements.txt
   ```

3. **Dataset file not found**:
   - Ensure `data/games.1000000.data` is in the correct folder.
   - Verify the path in `config.py`.

---

## Keeping Your Fork Up-to-Date (Optional)
If you’re working on a fork and want to sync with the original repository:
```bash
git remote add upstream https://github.com/Meddhif13/go-ai-project.git
git fetch upstream
git merge upstream/main
```

---

Happy coding! 🚀
