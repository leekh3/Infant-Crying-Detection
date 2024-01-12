import os
import pandas as pd
import glob

# Directory for batch files
batch_dir = "batch"
if not os.path.exists(batch_dir):
    os.makedirs(batch_dir)


# Load the Excel file
file_path = 'LENA_IDs_used_for_final_ADAA_analyses.xlsx'
df = pd.read_excel(file_path, engine='openpyxl')

# Assuming the IDs are in the first column, extract the first 100 IDs after the header
sdans = list(df.iloc[:100, 0])

# Name of your custom Conda environment
conda_env_name = "py38"

# Generate a batch file for each input
for sdan in sdans:
    batch_file_name = f"run_{sdan}.sh"
    batch_file_path = os.path.join(batch_dir, batch_file_name)
    input_wav = glob.glob(f"/data/NIMH_scratch/LENA_Recordings/100_selected_participants_use_me/{sdan}/*.wav")
    if len(input_wav) == 0:
        print(f"Input wav file does not exist:{sdan}")
        continue
    input_wav = input_wav[0]
    with open(batch_file_path, 'w') as file:
        file.write("#!/bin/bash\n\n")
        # file.write("#SBATCH --job-name=infant_cry_detection\n")
        # file.write("#SBATCH --gres=gpu:k80:1\n")
        # file.write("#SBATCH --cpus-per-task=8\n")
        # file.write("#SBATCH --mem=10G\n")
        # file.write("#SBATCH --output=infant_cry_detection_%j.out\n\n")
        # Conda initialization
        file.write("# >>> conda initialize >>>\n")
        file.write("# !! Contents within this block are managed by 'conda init' !!\n")
        file.write("__conda_setup=\"$('/data/leek13/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)\"\n")
        file.write("if [ $? -eq 0 ]; then\n")
        file.write("    eval \"$__conda_setup\"\n")
        file.write("else\n")
        file.write("    if [ -f \"/data/leek13/miniconda3/etc/profile.d/conda.sh\" ]; then\n")
        file.write("        . \"/data/leek13/miniconda3/etc/profile.d/conda.sh\"\n")
        file.write("    else\n")
        file.write("        export PATH=\"/data/leek13/miniconda3/bin:$PATH\"\n")
        file.write("    fi\n")
        file.write("fi\n")
        file.write("unset __conda_setup\n")
        file.write("# <<< conda initialize <<<\n\n")
        file.write(f"conda activate {conda_env_name}\n")  # Activate Conda environment
        file.write(f"python3 InfantCryDetectionPipeline_ACNP.py {input_wav} quality/{sdan}_quality.csv prediction/{sdan}_prediction.csv\n")

    # Submit the batch file to sbatch
    # os.system(f"sbatch {batch_file_path}")
    # os.system(f"sbatch  --partition=gpu --gres=gpu:k80:1 --gres=lscratch:20 --cpus-per-task=8 --ntasks=1 --mem=8G --job-name=$experiment_id --time=10-00:00 --wrap={batch_file_path}")
    os.system(f"sbatch --gres=lscratch:100 --cpus-per-task=2 batch/run_{sdan}.sh")
print("Batch files created and submitted.")
