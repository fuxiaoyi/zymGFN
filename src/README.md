# ProtRL

Protein sequence generation & optimization with RL/GRPO and structure evaluation.

---

## 1) Installation

Prereqs:
- Python ≥ 3.10
- Git
- CUDA-compatible GPU (recommended)
- (Optional) Conda/venv

```bash
git clone https://github.com/AI4PDLab/ProtRL.git
cd ProtRL
pip install -r requirements.txt
```

### GRPO-specific dependencies

Note: `torch>=2.6.0` is required for the GRPO trainer. Install the build that matches your CUDA.

Example (CUDA 12.1):
```bash
pip install --upgrade "torch>=2.6.0" --index-url https://download.pytorch.org/whl/cu121
```

Then install Python deps (pin TRL to 0.19.0):
```bash
pip install transformers pandas bitsandbytes peft evaluate \
            deepspeed accelerate tokenizers huggingface_hub requests \
            datasets matplotlib seaborn trl==0.19.0

conda install -c conda-forge -c bioconda foldseek
conda install -c conda-forge -c bioconda mmseqs2
```

---

## 2) External Models & Tools

You will need the following locally:
- ZymCTRL / EnzymCTRL model (base protein LM to fine-tune)
- facebook/esmfold_v1 (structure prediction)
- UniKP (kcat predictor)
- ToxinPred2 (toxicity predictor)

Prepare these models/tools per their docs and update local paths in the scripts.

---

## 3) Configuration

Update script variables to point to your local models/envs:

- In your `.sh` job scripts:
  - Replace the **model directory** (e.g., ZymCTRL/EnzymCTRL path)
  - Replace the **conda env name** (or venv path)

- In Python scripts, check default model paths:
  - `seq_gen.py`
  - `GRPO_train.py` (or your training entrypoint)
  - `ESM_Fold.py`

Each script has a `--model_dir` or in-file default; set these to your local directories.

---

## 3) Example

Using the `GRPO_backup/` folder as a runnable example. The end-to-end pipeline is scripted in **`GRPO.sh`**; open it to see each stage. In short:

1. **Train the policy**  
   Run `GRPO_train.py` to fine-tune the model. This creates an `output_iteration{N}/` directory with checkpoints.

2. **Sample sequences**  
   Run `seq_gen.py` to generate candidate protein sequences from the trained checkpoint.

3. **Fold structures & score similarity**  
   Run `ESM_Fold.py` to fold sequences with ESMFold and produce PDBs.  
   (If your script also calls Foldseek, this step computes TM-Score against the reference structure.)

4. **Compute auxiliary rewards**  
   Use **ToxinPred2** (toxicity) and **UniKP** (kcat) to score each sequence. These CSVs are later merged into the training set.

5. **Build the training dataset**  
   Run the dataset assembly script (e.g., `dataset_gen_*.py`) to combine TM-Score, toxicity, kcat, and length reward into per-example weights.

6. **Trainer implementations**  
   The **`src/`** directory contains shared utilities and trainer definitions (e.g., `GRPO`, `weightedDPO.py`, etc.) used by the pipeline.

> To reproduce the full loop, execute the shell script:
>
> ```bash
> cd GRPO_backup
> bash GRPO.sh
> # or, on a cluster with SLURM:
> # sbatch GRPO.sh
> ```
>
> The script iterates: **train → sample → fold/score → reward → dataset build → next iteration**.
