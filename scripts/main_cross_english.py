import argparse
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import os
from scipy.stats import spearmanr
from config import datasets_configs, average_config
import logging
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import glob
import re
from transformers import get_linear_schedule_with_warmup
from torch.nn import Dropout
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths for saving models and logs
log_dir = "/mnt/home/robinmeister/cross_english/logs"
model_dir = "/mnt/home/robinmeister/cross_english/models"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Function to save detailed logs
def save_logs(log_content, log_path):
    with open(log_path, 'a') as log_file:
        log_file.write(log_content + '\n')
    logger.info(f"Logs saved to {log_path}")

# Function to load datasets
def load_datasets(path_to_datasets):
    datasets = {
        "wmt2019_df": pd.read_csv(os.path.join(path_to_datasets, 'wmt2019_adversarial_dataset.csv')),
        "wmt2021_df": pd.read_csv(os.path.join(path_to_datasets, 'wmt2021_adversarial_dataset.csv')),
        "wmt2019_df_corrupted_random": pd.read_csv(os.path.join(path_to_datasets, 'wmt2019_adversarial_dataset_corrupted_0.04.csv')),
        "wmt2021_df_corrupted_random": pd.read_csv(os.path.join(path_to_datasets, 'wmt2021_adversarial_dataset_corrupted_0.04.csv')),
        "wmt2019_df_corrupted_ocr": pd.read_csv(os.path.join(path_to_datasets, 'wmt19_simple_ocr_noise.csv')),
        "wmt2021_df_corrupted_ocr": pd.read_csv(os.path.join(path_to_datasets, 'wmt21_simple_ocr_noise.csv')),
        "wmt2019_df_corrupted_blackletter": pd.read_csv(os.path.join(path_to_datasets, 'wmt19_blackletter_ocr_noise.csv')),
        "wmt2021_df_corrupted_blackletter": pd.read_csv(os.path.join(path_to_datasets, 'wmt21_blackletter_ocr_noise.csv')),
        "mono_df": pd.read_csv(os.path.join(path_to_datasets, 'mono_df.csv')),
        "cross_df": pd.read_csv(os.path.join(path_to_datasets, 'cross_df.csv')),
        "cross_en_de": pd.read_csv(os.path.join(path_to_datasets, 'cross_en_de.csv')),
        "cross_en_fr": pd.read_csv(os.path.join(path_to_datasets, 'cross_en_fr.csv')),
        "sts_es_en": pd.read_csv(os.path.join(path_to_datasets, 'sts17_es-en.csv')),
        "sts_ar_en": pd.read_csv(os.path.join(path_to_datasets, 'sts17_ar-en.csv')),
        "sts_tr_en": pd.read_csv(os.path.join(path_to_datasets, 'sts17_tr-en.csv')),
    }
    return datasets

def input_example_collate_fn(batch):
    texts1 = [example.texts[0] for example in batch]
    texts2 = [example.texts[1] for example in batch]
    labels = [example.label for example in batch]
    return texts1, texts2, labels

# Function to prepare training data
def prepare_training_data(df_de_fr, df_fr_de, df_en_fr, df_en_de, model_name, column_pairs):
    # Define training examples for each pair

    training_examples_de_fr = [
        InputExample(texts=[f"query: {row[source_col]}", f"query: {row[target_col]}"] if 'e5' in model_name else [row[source_col], row[target_col]], label=1.0)
        for _, row in df_de_fr.iterrows() for source_col, target_col in [column_pairs[0]]
    ]

    training_examples_fr_de = [
        InputExample(texts=[f"query: {row[source_col]}", f"query: {row[target_col]}"] if 'e5' in model_name else [row[source_col], row[target_col]], label=1.0)
        for _, row in df_fr_de.iterrows() for source_col, target_col in [column_pairs[1]]
    ]

    training_examples_en_fr = [
        InputExample(texts=[f"query: {row[source_col]}", f"query: {row[target_col]}"] if 'e5' in model_name else [row[source_col], row[target_col]], label=1.0)
        for _, row in df_en_fr.iterrows() for source_col, target_col in [column_pairs[2]]
    ]

    training_examples_en_de = [
        InputExample(texts=[f"query: {row[source_col]}", f"query: {row[target_col]}"] if 'e5' in model_name else [row[source_col], row[target_col]], label=1.0)
        for _, row in df_en_de.iterrows() for source_col, target_col in [column_pairs[3]]
    ]

    # Interleave the lists to ensure balanced batches
    balanced_examples = [
        example for sextet in zip(training_examples_de_fr, training_examples_fr_de, training_examples_en_fr, training_examples_en_de) for example in sextet
    ]

    return DataLoader(balanced_examples, batch_size=args.batch_size, shuffle=True, collate_fn=input_example_collate_fn)


# Function to train model and log loss
def train_model(model, loss_fn, train_dataloader, epochs, optimizer, scheduler, checkpoint_path, embedding_path, log_path, device, dropout_rate=0.1):
    logger.info(f"Training with loss function: {loss_fn}")
    
    if loss_fn == "CachedMultipleNegativesRankingLoss":
        train_loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=args.mini_batch_size)
    else:
        train_loss = getattr(losses, loss_fn)(model)

    # Initialize dropout
    dropout = Dropout(dropout_rate)

    # Initialize list to track loss
    batch_losses = []
    epoch_losses = []
    lrs = []

    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        total_loss = 0
        num_batches = 0

        for step, (texts1, texts2, labels) in enumerate(tqdm(train_dataloader, desc="Training Iterations")):
            # Log the batch texts for debugging
            logger.debug(f"Batch {step + 1}, Texts1: {texts1}, Texts2: {texts2}")

            # Tokenize the texts
            inputs1 = model.tokenize(texts1)
            inputs2 = model.tokenize(texts2)

            # Move inputs to the appropriate device
            inputs1 = {key: value.to(device) for key, value in inputs1.items()}
            inputs2 = {key: value.to(device) for key, value in inputs2.items()}

            # Forward pass
            outputs1 = model(inputs1)
            outputs2 = model(inputs2)

            # Get sentence embeddings
            embeddings1 = outputs1['sentence_embedding']
            embeddings2 = outputs2['sentence_embedding']

            # Apply dropout
            embeddings1 = dropout(embeddings1)
            embeddings2 = dropout(embeddings2)

            # Prepare sentence features
            sentence_features = [inputs1, inputs2]

            # Compute the loss
            labels = torch.ones(embeddings1.size(0), device=embeddings1.device)  # Assuming positive pairs, labels are 1
            loss = train_loss(sentence_features, labels)
            loss_value = loss.item()
            batch_losses.append(loss_value)
            logger.debug(f"Appended batch loss: {loss_value}")
            total_loss += loss_value
            num_batches += 1

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Adjust learning rate and log it
            scheduler.step()
            lrs.append(optimizer.param_groups[0]['lr'])
            logger.debug(f"Appended learning rate: {optimizer.param_groups[0]['lr']}")

            # Log the loss for the current batch
            logger.info(f"Epoch {epoch + 1}, Batch {step + 1}, Loss: {loss_value:.4f}")
            save_logs(f"Epoch {epoch + 1}, Batch {step + 1}, Loss: {loss_value:.4f}", log_path)

        # Compute average loss for the epoch
        avg_loss = total_loss / num_batches
        epoch_losses.append(avg_loss)
        logger.debug(f"Appended epoch loss: {avg_loss}")
        logger.info(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}")
        save_logs(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}", log_path)

        # Save model checkpoint
        #model.save(f"{checkpoint_path}")

    logger.debug(f"Batch losses: {batch_losses}")
    logger.debug(f"Epoch losses: {epoch_losses}")
    logger.debug(f"Learning rates: {lrs}")

    return model, batch_losses, epoch_losses, lrs

# Function to save embeddings for specific datasets
def save_specific_embeddings(model, model_name, datasets, dataset_names, save_path_prefix, device):
    prefix = "query: " if 'e5' in model_name else ""

    for dataset_name in dataset_names:
        df = datasets[dataset_name].head(10)  # Only take the first 10 rows

        if dataset_name == "sts_ar_en":
            df['en_embedding'] = df['en'].apply(
                lambda x: model.encode(prefix + str(x), convert_to_tensor=True, device=device).cpu().numpy().tolist())
            df['ar_embedding'] = df['ar'].apply(
                lambda x: model.encode(prefix + str(x), convert_to_tensor=True, device=device).cpu().numpy().tolist())

            df['similarity_en_ar'] = df.apply(lambda row: F.cosine_similarity(
                torch.tensor(row['en_embedding']).unsqueeze(0),
                torch.tensor(row['ar_embedding']).unsqueeze(0)
            ).item(), axis=1)

            save_path = f"{save_path_prefix}_{dataset_name}.csv"
            df.to_csv(save_path, index=False)
            logger.info(f"Embeddings for {dataset_name} saved at {save_path}")

        else:
            columns_to_embed = ['French', 'German', 'de_adv1', 'de_adv2', 'de_adv3', 'de_adv4', 'fr_adv1', 'fr_adv2',
                                'fr_adv3', 'fr_adv4']
            for column in columns_to_embed:
                df[f'{column}_emb'] = df[column].apply(
                    lambda x: model.encode(prefix + str(x), convert_to_tensor=True, device=device).cpu().numpy().tolist())

            similarity_columns = []

            # Calculate similarities between French and de_adv columns
            for i in range(1, 5):
                similarity_col = f'similarity_French_de_adv{i}'
                df[similarity_col] = df.apply(lambda row: F.cosine_similarity(
                    torch.tensor(row['French_emb']).unsqueeze(0),
                    torch.tensor(row[f'de_adv{i}_emb']).unsqueeze(0)
                ).item(), axis=1)
                similarity_columns.append(similarity_col)

            # Calculate similarities between German and fr_adv columns
            for i in range(1, 5):
                similarity_col = f'similarity_German_fr_adv{i}'
                df[similarity_col] = df.apply(lambda row: F.cosine_similarity(
                    torch.tensor(row['German_emb']).unsqueeze(0),
                    torch.tensor(row[f'fr_adv{i}_emb']).unsqueeze(0)
                ).item(), axis=1)
                similarity_columns.append(similarity_col)

            # Calculate similarity between French and German
            df['similarity_French_German'] = df.apply(lambda row: F.cosine_similarity(
                torch.tensor(row['French_emb']).unsqueeze(0),
                torch.tensor(row['German_emb']).unsqueeze(0)
            ).item(), axis=1)

            # Calculate similarity between German and French
            df['similarity_German_French'] = df.apply(lambda row: F.cosine_similarity(
                torch.tensor(row['German_emb']).unsqueeze(0),
                torch.tensor(row['French_emb']).unsqueeze(0)
            ).item(), axis=1)

            save_path = f"{save_path_prefix}_{dataset_name}.csv"
            df.to_csv(save_path, index=False)
            logger.info(f"Embeddings for {dataset_name} saved at {save_path}")

# plot metrics
def plot_and_save_metrics(batch_losses, epoch_losses, lrs, results_file, sample_sizes):
    plt.figure(figsize=(15, 5))

    # Plot batch losses
    plt.subplot(1, 3, 1)
    batch_idx = range(len(batch_losses))
    plt.plot(batch_idx, batch_losses, label='Batch Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Batch Losses')
    plt.legend()

    # Plot epoch losses
    plt.subplot(1, 3, 2)
    plt.plot(epoch_losses, label='Epoch Loss')
    plt.xlabel('Sample Size Index')
    plt.ylabel('Loss')
    plt.title('Epoch Losses')
    plt.xticks(range(len(sample_sizes)), sample_sizes)  # Display only the used sample sizes
    plt.legend()

    # Plot learning rates
    plt.subplot(1, 3, 3)
    plt.plot(range(len(lrs)), lrs, label='Learning Rate')  # Updated
    plt.xlabel('Sample Size Index')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rates')
    plt.xticks(range(len(sample_sizes)), sample_sizes)  # Display only the used sample sizes
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{results_file}_metrics.png")
    plt.close()

def evaluate_models(model_name, model, datasets, datasets_configs, device, batch_size=512):
    def cross_lingual_adversarial_evaluation(model_name, model, adv_df, source_column, target_column, adv_columns):
        prefix = "query: " if 'e5' in model_name else ""
        source_texts = [prefix + text for text in adv_df[source_column].fillna('Template Sentence').tolist()]
        target_texts = [prefix + text for text in adv_df[target_column].fillna('Template Sentence').tolist()]
        adv_texts = {idx: [prefix + text for text in adv_df[col].fillna('Template Sentence').tolist()] for idx, col in
                     enumerate(adv_columns)}

        def batch_encode(texts):
            all_embeddings = []
            dataloader = DataLoader(texts, batch_size=batch_size)
            for batch in tqdm(dataloader, desc="Encoding Batches"):
                embeddings = model.encode(batch, convert_to_tensor=True, batch_size=batch_size, device=device)
                all_embeddings.append(embeddings.cpu())
            return torch.cat(all_embeddings, dim=0)

        source_embeddings = F.normalize(batch_encode(source_texts), p=2, dim=1)
        target_embeddings = F.normalize(batch_encode(target_texts), p=2, dim=1)
        correct_translation_results = (source_embeddings @ target_embeddings.T).diag().numpy()

        result_df = pd.DataFrame({'Correct_Translation': correct_translation_results.round(3)})
        for idx in adv_texts:
            adv_embeddings = F.normalize(batch_encode(adv_texts[idx]), p=2, dim=1)
            adv_results = (source_embeddings @ adv_embeddings.T).diag().numpy()
            result_df[f'Adversarial{idx + 1}'] = adv_results.round(3)

        max_similarity = result_df.iloc[:, 1:].max(axis=1)
        accuracy = (result_df['Correct_Translation'] > max_similarity).mean() * 100
        return accuracy, result_df

    def evaluate_all_configs(model_name, model, datasets, datasets_configs):
        results = {'Model': model_name}
        for config in datasets_configs:
            logger.info(f"Evaluating configuration: {config['label']} with model {model_name}")
            dataset = datasets[config['dataset']]
            accuracy, _ = cross_lingual_adversarial_evaluation(
                model_name,
                model,
                dataset,
                config['source_column'],
                config['target_column'],
                config['adv_columns']
            )
            results[config['label']] = accuracy
        return pd.DataFrame([results])

    results_df = evaluate_all_configs(model_name, model, datasets, datasets_configs)
    return results_df

def evaluate_sts_task(model, sts_es_en, sts_ar_en, sts_tr_en, device, use_query_prefix=False):
    def add_prefix(texts, prefix):
        return [f"{prefix}{text}" for text in texts]

    # Evaluate on sts_es_en (es-en)
    sentences1_es_en = sts_es_en['es'].tolist()
    sentences2_es_en = sts_es_en['en'].tolist()
    if use_query_prefix:
        sentences1_es_en = add_prefix(sentences1_es_en, "query: ")
        sentences2_es_en = add_prefix(sentences2_es_en, "query: ")
    similarity_scores_es_en = sts_es_en['similarity_score'].tolist()
    embeddings1_es_en = model.encode(sentences1_es_en, convert_to_tensor=True, device=device)
    embeddings2_es_en = model.encode(sentences2_es_en, convert_to_tensor=True, device=device)
    cosine_scores_es_en = util.pytorch_cos_sim(embeddings1_es_en, embeddings2_es_en).diagonal().cpu().numpy()

    if len(cosine_scores_es_en) != len(similarity_scores_es_en):
        raise ValueError(f"Length mismatch: cosine_scores_es_en ({len(cosine_scores_es_en)}) vs similarity_scores_es_en ({len(similarity_scores_es_en)})")

    # Evaluate on sts_ar_en (ar-en)
    sentences1_ar_en = sts_ar_en['ar'].tolist()
    sentences2_ar_en = sts_ar_en['en'].tolist()
    if use_query_prefix:
        sentences1_ar_en = add_prefix(sentences1_ar_en, "query: ")
        sentences2_ar_en = add_prefix(sentences2_ar_en, "query: ")
    similarity_scores_ar_en = sts_ar_en['similarity_score'].tolist()
    embeddings1_ar_en = model.encode(sentences1_ar_en, convert_to_tensor=True, device=device)
    embeddings2_ar_en = model.encode(sentences2_ar_en, convert_to_tensor=True, device=device)
    cosine_scores_ar_en = util.pytorch_cos_sim(embeddings1_ar_en, embeddings2_ar_en).diagonal().cpu().numpy()

    if len(cosine_scores_ar_en) != len(similarity_scores_ar_en):
        raise ValueError(f"Length mismatch: cosine_scores_ar_en ({len(cosine_scores_ar_en)}) vs similarity_scores_ar_en ({len(similarity_scores_ar_en)})")

    # Evaluate on sts_tr_en (tr-en)
    sentences1_tr_en = sts_tr_en['tr'].tolist()
    sentences2_tr_en = sts_tr_en['en'].tolist()
    if use_query_prefix:
        sentences1_tr_en = add_prefix(sentences1_tr_en, "query: ")
        sentences2_tr_en = add_prefix(sentences2_tr_en, "query: ")
    similarity_scores_tr_en = sts_tr_en['similarity_score'].tolist()
    embeddings1_tr_en = model.encode(sentences1_tr_en, convert_to_tensor=True, device=device)
    embeddings2_tr_en = model.encode(sentences2_tr_en, convert_to_tensor=True, device=device)
    cosine_scores_tr_en = util.pytorch_cos_sim(embeddings1_tr_en, embeddings2_tr_en).diagonal().cpu().numpy()

    if len(cosine_scores_tr_en) != len(similarity_scores_tr_en):
        raise ValueError(f"Length mismatch: cosine_scores_tr_en ({len(cosine_scores_tr_en)}) vs similarity_scores_tr_en ({len(similarity_scores_tr_en)})")

    spearman_corr_es_en = spearmanr(cosine_scores_es_en, similarity_scores_es_en).correlation * 100
    spearman_corr_ar_en = spearmanr(cosine_scores_ar_en, similarity_scores_ar_en).correlation * 100
    spearman_corr_tr_en = spearmanr(cosine_scores_tr_en, similarity_scores_tr_en).correlation * 100

    avg_spearman_corr = (spearman_corr_es_en + spearman_corr_ar_en + spearman_corr_tr_en) / 3

    return spearman_corr_es_en, spearman_corr_ar_en, spearman_corr_tr_en, avg_spearman_corr

def add_average_columns_adjacent(df, config):
    for setting in config:
        cols = setting['columns']
        avg_col_name = f"{setting['label']}_avg"
        position = df.columns.get_loc(cols[-1]) + 1
        df.insert(position, avg_col_name, df[cols].mean(axis=1))
    return df

def main(args):
    logger.info("Loading datasets...")
    datasets = load_datasets(args.path_to_datasets)

    log_path = os.path.join(log_dir, f"{os.path.basename(args.results_file)}_training.log")

    all_batch_losses = []
    all_epoch_losses = []
    all_lrs = []

    sample_sizes = args.sample_sizes
    logger.info(f"Training with sample sizes: {sample_sizes}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SentenceTransformer(args.model_name).to(device)  # Initialize the model once

    # Calculate total training steps
    total_steps = 0
    sample_size_helper = 0
    for sample_size in sample_sizes:
        new_samples = sample_size - sample_size_helper
        num_full_batches = new_samples // args.batch_size
        remaining_samples = new_samples % args.batch_size
        new_training_steps = num_full_batches + (1 if remaining_samples > 0 else 0)
        total_steps += new_training_steps
        sample_size_helper = sample_size

    warmup_steps = int(0.1 * total_steps)
    logger.info(f"Training with total steps: {total_steps}")
    logger.info(f"Training with warmup_steps: {warmup_steps}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)

    start_idx = 0

    for sample_size in sample_sizes:
        logger.info(f"Training with sample size: {sample_size}")

        end_idx = sample_size

        # Split sample size equally between the six language pairs
        quarter_sample_size = (sample_size - start_idx) // 4

        column_pairs1 = [
            (args.source_column1_de, args.target_column1_fr),
            (args.source_column1_fr, args.target_column1_de),
            (args.source_column1_en, args.target_column1_fr),
            (args.source_column1_en, args.target_column1_de)
        ]

        df1_de_fr = datasets["cross_df"].iloc[start_idx:start_idx + quarter_sample_size]
        df1_fr_de = datasets["cross_df"].iloc[start_idx + quarter_sample_size:start_idx + 2 * quarter_sample_size]
        df1_en_fr = datasets["cross_en_fr"].iloc[start_idx:start_idx + quarter_sample_size]
        df1_en_de = datasets["cross_en_de"].iloc[start_idx + quarter_sample_size:start_idx + 2 * quarter_sample_size]

        combined_train_dataloader = prepare_training_data(df1_de_fr, df1_fr_de, df1_en_fr, df1_en_de, args.model_name, column_pairs1)

        logger.info(f"Training model on the combined datasets with sample size {sample_size}...")

        checkpoint_path = os.path.join(model_dir, 'intfloat' if 'e5' in args.model_name else 'sentence-transformers', f"{os.path.basename(args.results_file)}_checkpoint_sample{sample_size}.pt")
        embedding_path = os.path.join(model_dir, 'intfloat' if 'e5' in args.model_name else 'sentence-transformers', f"{os.path.basename(args.results_file)}_embeddings_sample{sample_size}")
        model, batch_losses, epoch_losses, lrs = train_model(model, args.loss_fn, combined_train_dataloader, args.epochs, optimizer, scheduler, checkpoint_path, embedding_path, log_path, device, dropout_rate=0.1)

        all_batch_losses.extend(batch_losses)
        all_epoch_losses.extend(epoch_losses)
        all_lrs.extend(lrs)

        logger.info(f"Evaluating model trained with sample size {sample_size}...")
        results_df = evaluate_models(args.model_name, model, datasets, datasets_configs, device=device)
        if args.sts:
            use_query_prefix = 'e5' in args.model_name
            sts_score_es_en, sts_score_ar_en, sts_score_tr_en, avg_sts_score = evaluate_sts_task(model,
                                                                                                 datasets["sts_es_en"],
                                                                                                 datasets["sts_ar_en"],
                                                                                                 datasets["sts_tr_en"],
                                                                                                 device=device,
                                                                                                 use_query_prefix=use_query_prefix)
            results_df['STS17_es_en'] = sts_score_es_en
            results_df['STS17_ar_en'] = sts_score_ar_en
            results_df['STS17_tr_en'] = sts_score_tr_en
            results_df['STS17_avg'] = avg_sts_score

        logger.info("Averaging results...")
        avg_results_df = add_average_columns_adjacent(results_df, average_config)
        logger.info(f"Averaged Results DataFrame:\n{avg_results_df}")

        avg_results_df.to_csv(f"{args.results_file}_sample{sample_size}.csv", index=False)
        logger.info(f"Results saved to {args.results_file}_sample{sample_size}.csv")

        specific_datasets = ["wmt2019_df", "wmt2019_df_corrupted_blackletter", "sts_ar_en"]
        save_path_prefix = f"{model_dir}/specific_embeddings_{args.model_name}_{args.loss_fn}_sample{sample_size}"
        save_specific_embeddings(model, args.model_name, datasets, specific_datasets, save_path_prefix, device)
        logger.info(f"Embeddings saved for sample size {sample_size} at {save_path_prefix}")

        start_idx = end_idx

    plot_and_save_metrics(all_batch_losses, all_epoch_losses, all_lrs, args.results_file, sample_sizes)
    logger.info(f"Metrics plot saved to {args.results_file}_metrics.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_datasets", type=str, required=True, help="Path to the datasets folder")
    parser.add_argument("--dataset1", type=str, required=True, help="Name of the first dataset to train on")
    parser.add_argument("--source_column1_de", type=str, required=True,
                        help="Name of the German source column in the first dataset")
    parser.add_argument("--target_column1_de", type=str, required=True,
                        help="Name of the German target column in the first dataset")
    parser.add_argument("--source_column1_fr", type=str, required=True,
                        help="Name of the French source column in the first dataset")
    parser.add_argument("--target_column1_fr", type=str, required=True,
                        help="Name of the French target column in the first dataset")
    parser.add_argument("--source_column1_en", type=str, required=True,
                        help="Name of the English source column in the first dataset")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to finetune")
    parser.add_argument("--loss_fn", type=str, required=True,
                        choices=["CosineSimilarityLoss", "MultipleNegativesRankingLoss", "CachedMultipleNegativesRankingLoss"], help="Loss function to use")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs for training")
    parser.add_argument("--sts", action="store_true", help="Evaluate on STS task")
    parser.add_argument("--sample_sizes", type=int, nargs='+', required=True, help="List of sample sizes to use for training")
    parser.add_argument("--results_file", type=str, required=True,
                        help="File to save the final averaged evaluation results")
    parser.add_argument("--mini_batch_size", type=int, default=8,
                        help="Mini-batch size for CachedMultipleNegativesRankingLoss")

    args = parser.parse_args()
    main(args)


