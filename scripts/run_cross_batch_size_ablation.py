import os
import itertools

# Define the combinations to test
datasets1 = ["cross_df"]
model_names = ["intfloat/multilingual-e5-base", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"]  # , "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
loss_functions = ["CachedMultipleNegativesRankingLoss"]  # "CosineSimilarityLoss", "MultipleNegativesRankingLoss"
batch_sizes = [12, 24, 48, 96]
epochs = [1]
sample_sizes = [15000]  # Define sample sizes to test

# Define source and target columns for each dataset
noise_rate1 = "005"

source_columns1_de = ["de"]
target_columns1_de = [f"de_{noise_rate1}"]
source_columns1_fr = ["fr"]
target_columns1_fr = [f"fr_{noise_rate1}"]

# Define paths
path_to_datasets = "/mnt/home/robinmeister/inputs"
results_dir = "/mnt/home/robinmeister/batch_size_ablation/results"

# Ensure results directory exists
os.makedirs(results_dir, exist_ok=True)

# Function to generate results file name based on parameters
def generate_results_file(dataset1, noise_rate1, model_name, loss_fn, batch_size, epochs):
    return f"{model_name}_{loss_fn}_cross_batches_ablation_bs{batch_size}_ep{epochs}.csv"

# Loop through all combinations
for (dataset1, model_name, loss_fn, batch_size, epoch,
     source_col1_de, target_col1_de, source_col1_fr, target_col1_fr) in itertools.product(
        datasets1, model_names, loss_functions, batch_sizes, epochs,
        source_columns1_de, target_columns1_de, source_columns1_fr, target_columns1_fr):

    results_file = os.path.join(results_dir, generate_results_file(dataset1, noise_rate1, model_name, loss_fn, batch_size, epoch))
    command = (
        f"python main_cross_batch_size_ablation.py --path_to_datasets {path_to_datasets} --dataset1 {dataset1} "
        f"--source_column1_de {source_col1_de} --target_column1_de {target_col1_de} "
        f"--source_column1_fr {source_col1_fr} --target_column1_fr {target_col1_fr} "
        f"--model_name {model_name} --loss_fn {loss_fn} "
        f"--batch_size {batch_size} --epochs {epoch} --results_file {results_file} "
        f"--mini_batch_size 8 --sts --sample_sizes {' '.join(map(str, sample_sizes))}"
    )

    print(f"Running: {command}")
    os.system(command)
