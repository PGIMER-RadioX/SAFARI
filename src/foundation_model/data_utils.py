import pandas as pd
from sklearn.impute import SimpleImputer

def load_radiomics_data(file_path, verbose=True):
    """
    Load radiomics data and separate features from metadata.
    This is the single source of truth for data loading.
    
    Args:
        file_path (str): Path to the radiomics CSV file.
        verbose (bool): Whether to print detailed information during loading.

    Returns:
        tuple: A tuple containing:
            - features (pd.DataFrame): The numerical feature data.
            - metadata (pd.DataFrame): The metadata.
    """
    df = pd.read_csv(file_path, low_memory=False)
    if verbose:
        print(f"  - Loaded dataset from {file_path}")
        print(f"  - Total columns: {len(df.columns)}")
        print(f"  - Total samples: {len(df)}")

    # Consolidated metadata identification logic
    metadata_patterns = [
        'patient', 'Patient', 'ID', 'id', 'mask', 'Mask',
        'subvolume', 'grid', 'position', 'slice', 'target',
        'Target', 'label', 'Label', 'class', 'Class', 'progression', 'if_nonv',
        'Histopathological_Grade', 'EGFR_mutation_status', 'KRAS_mutation_status', 'ALK_translocation_status', 'Recurrence',
        'Pathology', 'Vascular_invasion', 'Metastasis', 'Lymphnodes', 'PortalVeinThrombosis', 'AFP_group', 'TTP', 'Censored_0_progressed_1', 'OS'
    ]
    metadata_cols = []
    for col in df.columns:
        if any(pattern in col for pattern in metadata_patterns):
            metadata_cols.append(col)
        elif col.endswith('_i') or col.endswith('_j') or col.endswith('_k'):
            metadata_cols.append(col)

    if verbose:
        print(f"  - Identified {len(metadata_cols)} metadata columns.")

    metadata = df[metadata_cols].copy() if metadata_cols else pd.DataFrame(index=df.index)
    features = df.drop(columns=metadata_cols, errors='ignore')

    if verbose:
        print(f"  - Features shape: {features.shape}")
        print(f"  - Metadata shape: {metadata.shape}")

    # NaN Imputation Logic
    if features.isnull().values.any():
        if verbose:
            print(f"  - Detected {features.isnull().sum().sum()} NaN values in the features. Imputing with mean...")
        imputer = SimpleImputer(strategy='mean')
        features = pd.DataFrame(imputer.fit_transform(features), columns=features.columns)
        if verbose:
            print("  - NaN values successfully imputed.")
            
    return features, metadata