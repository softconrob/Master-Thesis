# Define the datasets_configs variable
datasets_configs = [
    # Original dataset 2019 & 2021 evaluations
    {"dataset": "wmt2019_df", "source_column": "French", "target_column": "German", "adv_columns": ["de_adv1", "de_adv2", "de_adv3", "de_adv4"], "label": "FR-DE 2019 Original"},
    {"dataset": "wmt2019_df", "source_column": "German", "target_column": "French", "adv_columns": ["fr_adv1", "fr_adv2", "fr_adv3", "fr_adv4"], "label": "DE-FR 2019 Original"},
    {"dataset": "wmt2021_df", "source_column": "French", "target_column": "German", "adv_columns": ["de_adv1", "de_adv2", "de_adv3", "de_adv4"], "label": "FR-DE 2021 Original"},
    {"dataset": "wmt2021_df", "source_column": "German", "target_column": "French", "adv_columns": ["fr_adv1", "fr_adv2", "fr_adv3", "fr_adv4"], "label": "DE-FR 2021 Original"},

    # Random...
    # Clean Source with Corrupted Target
    {"dataset": "wmt2019_df_corrupted_random", "source_column": "French_clean", "target_column": "German", "adv_columns": ["de_adv1", "de_adv2", "de_adv3", "de_adv4"], "label": "Random FR-DE 2019 Clean-Corrupted"},
    {"dataset": "wmt2019_df_corrupted_random", "source_column": "German_clean", "target_column": "French", "adv_columns": ["fr_adv1", "fr_adv2", "fr_adv3", "fr_adv4"], "label": "Random DE-FR 2019 Clean-Corrupted"},
    {"dataset": "wmt2021_df_corrupted_random", "source_column": "French_clean", "target_column": "German", "adv_columns": ["de_adv1", "de_adv2", "de_adv3", "de_adv4"], "label": "Random FR-DE 2021 Clean-Corrupted"},
    {"dataset": "wmt2021_df_corrupted_random", "source_column": "German_clean", "target_column": "French", "adv_columns": ["fr_adv1", "fr_adv2", "fr_adv3", "fr_adv4"], "label": "Random DE-FR 2021 Clean-Corrupted"},
    # Corrupted Source with Clean Target
    {"dataset": "wmt2019_df_corrupted_random", "source_column": "French", "target_column": "German_clean", "adv_columns": ["de_adv1", "de_adv2", "de_adv3", "de_adv4"], "label": "Random FR-DE 2019 Corrupted-Clean"},
    {"dataset": "wmt2019_df_corrupted_random", "source_column": "German", "target_column": "French_clean", "adv_columns": ["fr_adv1", "fr_adv2", "fr_adv3", "fr_adv4"], "label": "Random DE-FR 2019 Corrupted-Clean"},
    {"dataset": "wmt2021_df_corrupted_random", "source_column": "French", "target_column": "German_clean", "adv_columns": ["de_adv1", "de_adv2", "de_adv3", "de_adv4"], "label": "Random FR-DE 2021 Corrupted-Clean"},
    {"dataset": "wmt2021_df_corrupted_random", "source_column": "German", "target_column": "French_clean", "adv_columns": ["fr_adv1", "fr_adv2", "fr_adv3", "fr_adv4"], "label": "Random DE-FR 2021 Corrupted-Clean"},
    # Corrupted Source with Corrupted Target
    {"dataset": "wmt2019_df_corrupted_random", "source_column": "French", "target_column": "German", "adv_columns": ["de_adv1", "de_adv2", "de_adv3", "de_adv4"], "label": "Random FR-DE 2019 Corrupted-Corrupted"},
    {"dataset": "wmt2019_df_corrupted_random", "source_column": "German", "target_column": "French", "adv_columns": ["fr_adv1", "fr_adv2", "fr_adv3", "fr_adv4"], "label": "Random DE-FR 2019 Corrupted-Corrupted"},
    {"dataset": "wmt2021_df_corrupted_random", "source_column": "French", "target_column": "German", "adv_columns": ["de_adv1", "de_adv2", "de_adv3", "de_adv4"], "label": "Random FR-DE 2021 Corrupted-Corrupted"},
    {"dataset": "wmt2021_df_corrupted_random", "source_column": "German", "target_column": "French", "adv_columns": ["fr_adv1", "fr_adv2", "fr_adv3", "fr_adv4"], "label": "Random DE-FR 2021 Corrupted-Corrupted"},

    # OCR...
    # Clean Source with Corrupted Target
    {"dataset": "wmt2019_df_corrupted_ocr", "source_column": "French_clean", "target_column": "German", "adv_columns": ["de_adv1", "de_adv2", "de_adv3", "de_adv4"], "label": "OCR FR-DE 2019 Clean-Corrupted"},
    {"dataset": "wmt2019_df_corrupted_ocr", "source_column": "German_clean", "target_column": "French", "adv_columns": ["fr_adv1", "fr_adv2", "fr_adv3", "fr_adv4"], "label": "OCR DE-FR 2019 Clean-Corrupted"},
    {"dataset": "wmt2021_df_corrupted_ocr", "source_column": "French_clean", "target_column": "German", "adv_columns": ["de_adv1", "de_adv2", "de_adv3", "de_adv4"], "label": "OCR FR-DE 2021 Clean-Corrupted"},
    {"dataset": "wmt2021_df_corrupted_ocr", "source_column": "German_clean", "target_column": "French", "adv_columns": ["fr_adv1", "fr_adv2", "fr_adv3", "fr_adv4"], "label": "OCR DE-FR 2021 Clean-Corrupted"},
    # Corrupted Source with Clean Target
    {"dataset": "wmt2019_df_corrupted_ocr", "source_column": "French", "target_column": "German_clean", "adv_columns": ["de_adv1", "de_adv2", "de_adv3", "de_adv4"], "label": "OCR FR-DE 2019 Corrupted-Clean"},
    {"dataset": "wmt2019_df_corrupted_ocr", "source_column": "German", "target_column": "French_clean", "adv_columns": ["fr_adv1", "fr_adv2", "fr_adv3", "fr_adv4"], "label": "OCR DE-FR 2019 Corrupted-Clean"},
    {"dataset": "wmt2021_df_corrupted_ocr", "source_column": "French", "target_column": "German_clean", "adv_columns": ["de_adv1", "de_adv2", "de_adv3", "de_adv4"], "label": "OCR FR-DE 2021 Corrupted-Clean"},
    {"dataset": "wmt2021_df_corrupted_ocr", "source_column": "German", "target_column": "French_clean", "adv_columns": ["fr_adv1", "fr_adv2", "fr_adv3", "fr_adv4"], "label": "OCR DE-FR 2021 Corrupted-Clean"},
    # Corrupted Source with Corrupted Target
    {"dataset": "wmt2019_df_corrupted_ocr", "source_column": "French", "target_column": "German", "adv_columns": ["de_adv1", "de_adv2", "de_adv3", "de_adv4"], "label": "OCR FR-DE 2019 Corrupted-Corrupted"},
    {"dataset": "wmt2019_df_corrupted_ocr", "source_column": "German", "target_column": "French", "adv_columns": ["fr_adv1", "fr_adv2", "fr_adv3", "fr_adv4"], "label": "OCR DE-FR 2019 Corrupted-Corrupted"},
    {"dataset": "wmt2021_df_corrupted_ocr", "source_column": "French", "target_column": "German", "adv_columns": ["de_adv1", "de_adv2", "de_adv3", "de_adv4"], "label": "OCR FR-DE 2021 Corrupted-Corrupted"},
    {"dataset": "wmt2021_df_corrupted_ocr", "source_column": "German", "target_column": "French", "adv_columns": ["fr_adv1", "fr_adv2", "fr_adv3", "fr_adv4"], "label": "OCR DE-FR 2021 Corrupted-Corrupted"},

    # Blackletter...
    # Clean Source with Corrupted Target
    {"dataset": "wmt2019_df_corrupted_blackletter", "source_column": "French_clean", "target_column": "German", "adv_columns": ["de_adv1", "de_adv2", "de_adv3", "de_adv4"], "label": "Blackletter FR-DE 2019 Clean-Corrupted"},
    {"dataset": "wmt2019_df_corrupted_blackletter", "source_column": "German_clean", "target_column": "French", "adv_columns": ["fr_adv1", "fr_adv2", "fr_adv3", "fr_adv4"], "label": "Blackletter DE-FR 2019 Clean-Corrupted"},
    {"dataset": "wmt2021_df_corrupted_blackletter", "source_column": "French_clean", "target_column": "German", "adv_columns": ["de_adv1", "de_adv2", "de_adv3", "de_adv4"], "label": "Blackletter FR-DE 2021 Clean-Corrupted"},
    {"dataset": "wmt2021_df_corrupted_blackletter", "source_column": "German_clean", "target_column": "French", "adv_columns": ["fr_adv1", "fr_adv2", "fr_adv3", "fr_adv4"], "label": "Blackletter DE-FR 2021 Clean-Corrupted"},
    # Corrupted Source with Clean Target
    {"dataset": "wmt2019_df_corrupted_blackletter", "source_column": "French", "target_column": "German_clean", "adv_columns": ["de_adv1", "de_adv2", "de_adv3", "de_adv4"], "label": "Blackletter FR-DE 2019 Corrupted-Clean"},
    {"dataset": "wmt2019_df_corrupted_blackletter", "source_column": "German", "target_column": "French_clean", "adv_columns": ["fr_adv1", "fr_adv2", "fr_adv3", "fr_adv4"], "label": "Blackletter DE-FR 2019 Corrupted-Clean"},
    {"dataset": "wmt2021_df_corrupted_blackletter", "source_column": "French", "target_column": "German_clean", "adv_columns": ["de_adv1", "de_adv2", "de_adv3", "de_adv4"], "label": "Blackletter FR-DE 2021 Corrupted-Clean"},
    {"dataset": "wmt2021_df_corrupted_blackletter", "source_column": "German", "target_column": "French_clean", "adv_columns": ["fr_adv1", "fr_adv2", "fr_adv3", "fr_adv4"], "label": "Blackletter DE-FR 2021 Corrupted-Clean"},
    # Corrupted Source with Corrupted Target
    {"dataset": "wmt2019_df_corrupted_blackletter", "source_column": "French", "target_column": "German", "adv_columns": ["de_adv1", "de_adv2", "de_adv3", "de_adv4"], "label": "Blackletter FR-DE 2019 Corrupted-Corrupted"},
    {"dataset": "wmt2019_df_corrupted_blackletter", "source_column": "German", "target_column": "French", "adv_columns": ["fr_adv1", "fr_adv2", "fr_adv3", "fr_adv4"], "label": "Blackletter DE-FR 2019 Corrupted-Corrupted"},
    {"dataset": "wmt2021_df_corrupted_blackletter", "source_column": "French", "target_column": "German", "adv_columns": ["de_adv1", "de_adv2", "de_adv3", "de_adv4"], "label": "Blackletter FR-DE 2021 Corrupted-Corrupted"},
    {"dataset": "wmt2021_df_corrupted_blackletter", "source_column": "German", "target_column": "French", "adv_columns": ["fr_adv1", "fr_adv2", "fr_adv3", "fr_adv4"], "label": "Blackletter DE-FR 2021 Corrupted-Corrupted"},
    # others...
]

# Define the config for average computation
average_config = [
    {'label': 'avg_original', 'columns': ['FR-DE 2019 Original', 'DE-FR 2019 Original', 'FR-DE 2021 Original', 'DE-FR 2021 Original']},

    # Random
    {'label': 'random_avg_clean_corrputed', 'columns': ['Random FR-DE 2019 Clean-Corrupted', 'Random DE-FR 2019 Clean-Corrupted', 'Random FR-DE 2021 Clean-Corrupted', 'Random DE-FR 2021 Clean-Corrupted']},
    {'label': 'random_avg_corrupted_clean', 'columns': ['Random FR-DE 2019 Corrupted-Clean', 'Random DE-FR 2019 Corrupted-Clean', 'Random FR-DE 2021 Corrupted-Clean', 'Random DE-FR 2021 Corrupted-Clean']},
    {'label': 'random_avg_corrupted_corrupted', 'columns': ['Random FR-DE 2019 Corrupted-Corrupted', 'Random DE-FR 2019 Corrupted-Corrupted', 'Random FR-DE 2021 Corrupted-Corrupted', 'Random DE-FR 2021 Corrupted-Corrupted']},

    # OCR
    {'label': 'ocr_avg_clean_corrputed', 'columns': ['OCR FR-DE 2019 Clean-Corrupted', 'OCR DE-FR 2019 Clean-Corrupted', 'OCR FR-DE 2021 Clean-Corrupted', 'OCR DE-FR 2021 Clean-Corrupted']},
    {'label': 'ocr_avg_corrupted_clean', 'columns': ['OCR FR-DE 2019 Corrupted-Clean', 'OCR DE-FR 2019 Corrupted-Clean', 'OCR FR-DE 2021 Corrupted-Clean', 'OCR DE-FR 2021 Corrupted-Clean']},
    {'label': 'ocr_avg_corrupted_corrupted', 'columns': ['OCR FR-DE 2019 Corrupted-Corrupted', 'OCR DE-FR 2019 Corrupted-Corrupted', 'OCR FR-DE 2021 Corrupted-Corrupted', 'OCR DE-FR 2021 Corrupted-Corrupted']},

    # Blackletter
    {'label': 'blackletter_avg_clean_corrputed', 'columns': ['Blackletter FR-DE 2019 Clean-Corrupted', 'Blackletter DE-FR 2019 Clean-Corrupted', 'Blackletter FR-DE 2021 Clean-Corrupted', 'Blackletter DE-FR 2021 Clean-Corrupted']},
    {'label': 'blackletter_avg_corrupted_clean', 'columns': ['Blackletter FR-DE 2019 Corrupted-Clean', 'Blackletter DE-FR 2019 Corrupted-Clean', 'Blackletter FR-DE 2021 Corrupted-Clean', 'Blackletter DE-FR 2021 Corrupted-Clean']},
    {'label': 'blackletter_avg_corrupted_corrupted', 'columns': ['Blackletter FR-DE 2019 Corrupted-Corrupted', 'Blackletter DE-FR 2019 Corrupted-Corrupted', 'Blackletter FR-DE 2021 Corrupted-Corrupted', 'Blackletter DE-FR 2021 Corrupted-Corrupted']},
    # others...
]