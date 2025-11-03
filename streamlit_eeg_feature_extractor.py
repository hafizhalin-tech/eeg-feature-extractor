import streamlit as st
import pandas as pd
import numpy as np
import os
from scipy.signal import butter, filtfilt, welch
from scipy.stats import skew, kurtosis, entropy
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_classif

# -----------------------------
# Streamlit App Configuration
# -----------------------------
st.set_page_config(page_title="EEG Feature Extractor", layout="wide")
st.title("🧠 EEG Multi-File Feature Extraction and MI Analysis")

# -----------------------------
# Helper Functions
# -----------------------------
def butter_filter(data, filter_type, cutoff, fs=128, order=5):
    nyq = 0.5 * fs
    normal_cutoff = np.array(cutoff) / nyq
    b, a = butter(order, normal_cutoff, btype=filter_type, analog=False)
    return filtfilt(b, a, data)

def segment_signal(data, segment_length, overlap):
    step = segment_length - overlap
    segments = []
    for start in range(0, len(data) - segment_length + 1, step):
        segments.append(data[start:start + segment_length])
    return np.array(segments)

def extract_features(segment):
    # Time-domain features
    mean_val = np.mean(segment)
    std_val = np.std(segment)
    skew_val = skew(segment)
    kurt_val = kurtosis(segment)
    entropy_val = entropy(np.abs(segment))
    
    # Frequency-domain (using Welch)
    f, pxx = welch(segment, fs=128)
    band_powers = [np.sum(pxx[(f >= low) & (f < high)]) 
                   for low, high in [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 45)]]
    
    # Combine features
    return [mean_val, std_val, skew_val, kurt_val, entropy_val] + band_powers

def process_file(df, filter_type, normalize, segment_len, overlap):
    features = []
    channels = df.columns

    for ch in channels:
        signal = df[ch].values

        # Apply filter
        if filter_type == "Low-pass":
            signal = butter_filter(signal, 'low', [30])
        elif filter_type == "High-pass":
            signal = butter_filter(signal, 'high', [1])
        elif filter_type == "Band-pass":
            signal = butter_filter(signal, 'band', [1, 30])

        # Normalize
        if normalize:
            signal = MinMaxScaler().fit_transform(signal.reshape(-1, 1)).flatten()

        # Segment
        segments = segment_signal(signal, segment_len, overlap)
        for seg in segments:
            feat = extract_features(seg)
            features.append(feat)

    feature_names = ["Mean", "Std", "Skew", "Kurtosis", "Entropy",
                     "Delta", "Theta", "Alpha", "Beta", "Gamma"]
    columns = [f"{ch}_{f}" for ch in channels for f in feature_names]
    return pd.DataFrame([np.concatenate(features)], columns=columns)

# -----------------------------
# App UI Controls
# -----------------------------
st.sidebar.header("⚙️ Processing Settings")

uploaded_files = st.sidebar.file_uploader(
    "Upload multiple EEG CSV/XLSX files", 
    type=["csv", "xlsx"], 
    accept_multiple_files=True
)

filter_type = st.sidebar.selectbox(
    "Select filter type", 
    ["None", "Low-pass", "High-pass", "Band-pass"]
)

normalize = st.sidebar.checkbox("Normalize signals", value=True)

segment_len = st.sidebar.number_input(
    "Segment length (samples)", 
    min_value=64, 
    max_value=2048, 
    value=256
)

overlap = st.sidebar.number_input(
    "Overlap (samples)", 
    min_value=0, 
    max_value=2048, 
    value=64
)

# -----------------------------
# MI Threshold Slider (Persistent)
# -----------------------------
if "mi_threshold" not in st.session_state:
    st.session_state.mi_threshold = 0.05

st.sidebar.markdown("### 🧠 MI Feature Selection")
st.sidebar.write("Adjust the Mutual Information (MI) threshold below:")

st.session_state.mi_threshold = st.sidebar.slider(
    "MI Threshold",
    min_value=0.0,
    max_value=1.0,
    value=st.session_state.mi_threshold,
    step=0.01,
)

# -----------------------------
# Processing and Feature Extraction
# -----------------------------
if uploaded_files:
    all_features = []
    labels = []

    st.write("### Uploaded Files")
    for f in uploaded_files:
        st.write(f"✅ {f.name}")

        if f.name.endswith(".csv"):
            df = pd.read_csv(f)
        else:
            df = pd.read_excel(f)

        st.write(f"Processing **{f.name}** ...")

        features_df = process_file(df, filter_type, normalize, segment_len, overlap)
        features_df["Filename"] = f.name
        all_features.append(features_df)

    all_data = pd.concat(all_features, ignore_index=True)

    # -----------------------------
    # MI Computation
    # -----------------------------
    st.subheader("📊 Mutual Information (Feature Importance)")
    st.write("Computing MI scores...")

    # Example label (use filenames as pseudo labels for demo)
    y = np.arange(len(all_data))
    X = all_data.drop(columns=["Filename"])

    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_df = pd.DataFrame({"Feature": X.columns, "MI_Score": mi_scores})
    mi_df = mi_df.sort_values("MI_Score", ascending=False)

    # Apply threshold
    selected = mi_df[mi_df["MI_Score"] >= st.session_state.mi_threshold]

    st.write(f"### Features above threshold ({st.session_state.mi_threshold:.2f})")
    st.dataframe(selected)

    st.bar_chart(selected.set_index("Feature"))

    # -----------------------------
    # Save Option
    # -----------------------------
    if not selected.empty:
        if st.button("💾 Save selected features to CSV"):
            save_path = "selected_features.csv"
            selected.to_csv(save_path, index=False)
            st.success(f"File saved as `{save_path}` in app directory.")
    else:
        st.warning("No features exceed the current MI threshold.")

else:
    st.info("👆 Please upload EEG files to begin processing.")

st.markdown("---")
st.caption("Developed for multi-channel EEG feature extraction and MI-based feature ranking.")
