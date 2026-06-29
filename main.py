import pandas as pd
import glob
import os
import re
from scipy.signal import butter, filtfilt
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# ---------- Load individual .txt file ---------- 
def load_new_format_txt(file_path):
    """
    Loads angular momentum data from a .txt file into WBAM and segment DataFrames.
    Returns wbam_df and segment_df with standardized structure.
    """
    filename = os.path.basename(file_path)
    match = re.match(r"(pp\d+)_(\d+)\.txt", filename)
    if not match:
        raise ValueError("Filename must be like 'ppX_00Y.txt'")
    participant, trial = match.groups()
    trial = int(trial.lstrip("0"))

    df = pd.read_csv(file_path, skiprows=4, sep="\t")
    df.dropna(axis=1, how="all", inplace=True)
    df.rename(columns={df.columns[0]: "Frame"}, inplace=True)

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    segment_labels = lines[1].strip().split("\t")[1:]  # skip "ITEM"

    reshaped_data = []
    for i in range(0, len(segment_labels), 3):
        raw_label = segment_labels[i]
        segment = raw_label.replace("Normalized_", "")
        if raw_label == "NORMALIZED_WBAM":
            segment = "WBAM"

        x_col, y_col, z_col = df.columns[i + 1], df.columns[i + 2], df.columns[i + 3]
        temp_df = df[["Frame", x_col, y_col, z_col]].copy()
        temp_df.columns = ["Frame", "X", "Y", "Z"]
        temp_df["Participant"] = participant
        temp_df["Trial"] = trial
        temp_df["Segment"] = segment
        reshaped_data.append(temp_df)

    full_df = pd.concat(reshaped_data, ignore_index=True)
    wbam_df = full_df[full_df["Segment"] == "WBAM"].copy()
    segment_df = full_df[full_df["Segment"] != "WBAM"].copy()

    return wbam_df, segment_df


# ---------- Load all .txt files in a folder ----------
def load_all_txt_files(folder_path):
    """
    Loads all .txt files and returns merged wbam_df and segment_df.
    """
    txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
    all_wbam = []
    all_segments = []

    for file_path in txt_files:
        try:
            wbam, segments = load_new_format_txt(file_path)
            all_wbam.append(wbam)
            all_segments.append(segments)
        except Exception as e:
            print(f"Error with {file_path}: {e}")

    wbam_df = pd.concat(all_wbam, ignore_index=True)
    segment_df = pd.concat(all_segments, ignore_index=True)

    return wbam_df, segment_df


# ---------- Set folder path and run ----------
# Replace this with your local folder path
folder_path = "Sidestep angular momentum data files"
wbam_df, segment_df = load_all_txt_files(folder_path)


events_df = pd.read_csv("Sidestep angular momentum data files/Computation_relavants.csv")

PREPARATION_FRAMES = 60
RECOVERY_FRAMES = 60

def assign_sidestep_phase(row, event_data):
    """
    Determines the phase (Preparation, Sidestep, Recovery) for each frame.
    
    Parameters:
    - row: The current row in WBAM or segment data.
    - event_data: DataFrame containing event frames for the participant & trial.

    Returns:
    - String indicating the movement phase ("Preparation", "Sidestep", "Recovery", "Outside")
    """
    participant = row["Participant"]
    trial = row["Trial"]
    frame = row["Frame"]

    event_row = event_data[(event_data["Participant"] == participant) & (event_data["Trial"] == trial)]

    if event_row.empty:
        return "Outside" 

    footstrike_ic = int(event_row["FootStrike_IC"].values[0])
    footoff_opp = int(event_row["FootOff_OPP"].values[0])

    if frame < footstrike_ic - PREPARATION_FRAMES:
        return "Outside" 
    elif footstrike_ic - PREPARATION_FRAMES <= frame < footstrike_ic:
        return "Preparation"
    elif footstrike_ic <= frame <= footoff_opp:
        return "Sidestep"
    elif footoff_opp < frame <= footoff_opp + RECOVERY_FRAMES:
        return "Recovery"
    else:
        return "Outside" 

wbam_df["Phase"] = wbam_df.apply(lambda row: assign_sidestep_phase(row, events_df), axis=1)
segment_df["Phase"] = segment_df.apply(lambda row: assign_sidestep_phase(row, events_df), axis=1)

# Count frames per phase in WBAM data
phase_counts = (
    wbam_df.groupby(["Participant", "Trial", "Phase"])["Frame"]
    .count()
    .unstack(fill_value=0)  
    .reset_index()
)

# Compute total frames across all three phases
phase_counts["Total_Frames"] = (
    phase_counts.get("Preparation", 0) + 
    phase_counts.get("Sidestep", 0) + 
    phase_counts.get("Recovery", 0)
)

max_sidestep_frames = phase_counts["Sidestep"].max()
print(
    f"Max number of frames for the sidestep phase is {max_sidestep_frames}"
    )

CUTOFF_FREQ = 6
SAMPLING_RATE = 60
FILTER_ORDER = 4

def butter_lowpass_filter(data, cutoff=CUTOFF_FREQ, fs=SAMPLING_RATE, order=FILTER_ORDER):
    """
    Apply a Butterworth low-pass filter to the given data.
    
    Parameters:
    - data: The signal to filter (numpy array).
    - cutoff: The cutoff frequency in Hz.
    - fs: Sampling frequency (frame rate), typically 60 Hz.
    - order: Order of the Butterworth filter.
    
    Returns:
    - Filtered signal (numpy array).
    """
    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyquist  # Convert to Nyquist domain
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)  # Apply zero-phase filtering

# remove all NA values that are not in the sidestep phase
wbam_df_mask = (~pd.isnull(wbam_df).any(axis=1)) | (wbam_df["Phase"] == "Sidestep")
segment_df_mask = (~pd.isnull(segment_df).any(axis=1)) | (segment_df["Phase"] == "Sidestep")
wbam_df_filt = wbam_df[wbam_df_mask].copy()
segment_df_filt = segment_df[segment_df_mask].copy()

assert not pd.isnull(wbam_df_filt).any().any()
assert not pd.isnull(segment_df_filt).any().any()


wbam_df_filt[["X", "Y", "Z"]] = wbam_df_filt.groupby(["Participant", "Trial"])[["X", "Y", "Z"]].transform(
    lambda col: butter_lowpass_filter(col.to_numpy())
)

segment_df_filt[["X", "Y", "Z"]] = segment_df_filt.groupby(["Participant", "Trial", "Segment"])[["X", "Y", "Z"]].transform(
    lambda col: butter_lowpass_filter(col.to_numpy())
)

TARGET_SIDESTEP_FRAMES = max_sidestep_frames

def extract_and_interpolate(df):
    """
    Extracts the three movement phases and interpolates the Sidestep phase to a fixed number of frames.

    Parameters:
    - df: DataFrame containing a single trial's data
    
    Returns:
    - DataFrame with combined original Preparation, interpolated Sidestep, and Recovery phases
    """
    preparation_df = df[df["Phase"] == "Preparation"].copy()
    sidestep_df = df[df["Phase"] == "Sidestep"].copy()
    recovery_df = df[df["Phase"] == "Recovery"].copy()

    if len(preparation_df) != PREPARATION_FRAMES or len(recovery_df) != RECOVERY_FRAMES:
        print(f"Warning: Incorrect frame count for {df['Participant'].iloc[0]}, Trial {df['Trial'].iloc[0]}")
        return None  

    if sidestep_df.empty:
        print(f"Warning: No sidestep data for {df['Participant'].iloc[0]}, Trial {df['Trial'].iloc[0]}")
        return None  

    original_frames = np.linspace(0, len(sidestep_df) - 1, len(sidestep_df))
    target_frames = np.linspace(0, len(sidestep_df) - 1, TARGET_SIDESTEP_FRAMES)

    interp_x = interp1d(original_frames, sidestep_df["X"], kind="linear")
    interp_y = interp1d(original_frames, sidestep_df["Y"], kind="linear")
    interp_z = interp1d(original_frames, sidestep_df["Z"], kind="linear")

    sidestep_interp_df = pd.DataFrame({
        "Frame": np.arange(PREPARATION_FRAMES, PREPARATION_FRAMES + TARGET_SIDESTEP_FRAMES),
        "X": interp_x(target_frames),
        "Y": interp_y(target_frames),
        "Z": interp_z(target_frames),
        "Participant": sidestep_df["Participant"].iloc[0],
        "Trial": sidestep_df["Trial"].iloc[0],
        "Segment": sidestep_df["Segment"].iloc[0],
        "Phase": "Sidestep"
    })

    preparation_df["Frame"] = np.arange(0, PREPARATION_FRAMES)
    recovery_df["Frame"] = np.arange(PREPARATION_FRAMES + TARGET_SIDESTEP_FRAMES, PREPARATION_FRAMES + TARGET_SIDESTEP_FRAMES + RECOVERY_FRAMES)
    combined_df = pd.concat([preparation_df, sidestep_interp_df, recovery_df], ignore_index=True)

    return combined_df

wbam_interpolated = wbam_df_filt.groupby(["Participant", "Trial"]).apply(extract_and_interpolate).reset_index(drop=True)
segment_interpolated = segment_df_filt.groupby(["Participant", "Trial", "Segment"]).apply(extract_and_interpolate).reset_index(drop=True)
print(wbam_interpolated.tail())



def get_participant_averaged_values(df):
    meaned_signals = []
    phases = ["Preparation", "Sidestep", "Recovery"]
    for participant in df["Participant"].unique():
        temp_df = df[(df["Participant"]==participant) & (df["Phase"].isin(phases))]
        mean_signal = temp_df.groupby(["Frame", "Segment"])[["X", "Y", "Z"]].mean().reset_index()
        combined = pd.merge(mean_signal, temp_df[["Phase", "Frame", "Participant", "Segment"]], on=["Frame", "Segment"], how="left").drop_duplicates()
        meaned_signals.append(combined)
    
    return pd.concat(meaned_signals, axis=0, ignore_index=True)

wbam_averaged = get_participant_averaged_values(wbam_interpolated)
segment_averaged = get_participant_averaged_values(segment_interpolated)



events_df["Stance_IC"] = events_df["FootOff_IC"] - events_df["FootStrike_IC"]
events_df["Stance_OPP"] = events_df["FootOff_OPP"] - events_df["FootStrike_OPP"]
events_df["Sidestep_Duration"] = events_df["FootOff_OPP"] - events_df["FootStrike_IC"]

events_df["Stance_IC_%"] = (events_df["Stance_IC"] / events_df["Sidestep_Duration"]) * 100
events_df["Stance_OPP_%"] = (events_df["Stance_OPP"] / events_df["Sidestep_Duration"]) * 100

avg_stance_ic_pct = events_df["Stance_IC_%"].mean()
avg_stance_opp_pct = events_df["Stance_OPP_%"].mean()

print(f"\nAverage % of Sidestep spent in Stance_IC (initial contact foot): {avg_stance_ic_pct:.1f}%")
print(f"Average % of Sidestep spent in Stance_OPP (opposite foot): {avg_stance_opp_pct:.1f}%")


# def plot_wbam_mean_sd(wbam_data, participant):
#     """
#     Plots the mean ± SD WBAM data for X, Y, and Z axes with movement phases.

#     Parameters:
#     - wbam_data: DataFrame containing WBAM data for all trials (interpolated)
#     - participant: The participant ID to plot
#     """
#     participant_data = wbam_data[wbam_data["Participant"] == participant]

#     mean_df = participant_data.groupby(["Frame", "Phase"])[["X", "Y", "Z"]].mean().reset_index()
#     std_df = participant_data.groupby(["Frame", "Phase"])[["X", "Y", "Z"]].std().reset_index()

#     SAMPLING_RATE = 60
#     FOOTSTRIKE_IC_FRAME = PREPARATION_FRAMES
#     FOOTOFF_OPP_FRAME = PREPARATION_FRAMES + TARGET_SIDESTEP_FRAMES

#     footstrike_ic_time = FOOTSTRIKE_IC_FRAME / SAMPLING_RATE
#     footoff_opp_time = FOOTOFF_OPP_FRAME / SAMPLING_RATE

#     stance_ic_offset = avg_stance_ic_pct / 100 * TARGET_SIDESTEP_FRAMES
#     footoff_ic_frame = FOOTSTRIKE_IC_FRAME + stance_ic_offset
#     footoff_ic_time = footoff_ic_frame / SAMPLING_RATE

#     stance_opp_offset = avg_stance_opp_pct / 100 * TARGET_SIDESTEP_FRAMES
#     footstrike_opp_frame = FOOTOFF_OPP_FRAME - stance_opp_offset
#     footstrike_opp_time = footstrike_opp_frame / SAMPLING_RATE

#     phase_colors = {
#         "Preparation": "tab:blue",
#         "Sidestep": "tab:orange",
#         "Recovery": "tab:green"
#     }

#     axis_labels = {
#         "X": "Normalized WBAM (X-axis, Frontal Plane)",
#         "Y": "Normalized WBAM (Y-axis, Sagittal Plane)",
#         "Z": "Normalized WBAM (Z-axis, Transverse Plane)"
#     }

#     fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

#     for ax, axis in zip(axes, ["X", "Y", "Z"]):
#         for phase, color in phase_colors.items():
#             phase_mask = mean_df["Phase"] == phase  
#             ax.plot(mean_df.loc[phase_mask, "Frame"] / SAMPLING_RATE, 
#                     mean_df.loc[phase_mask, axis], 
#                     label=f"{phase} - Mean", color=color)
            
#             ax.fill_between(mean_df.loc[phase_mask, "Frame"] / SAMPLING_RATE, 
#                             mean_df.loc[phase_mask, axis] - std_df.loc[phase_mask, axis], 
#                             mean_df.loc[phase_mask, axis] + std_df.loc[phase_mask, axis], 
#                             color=color, alpha=0.2, label=f"{phase} ±1 SD")

#         line1 = ax.axvline(footstrike_ic_time, color="red", linestyle="solid", linewidth=1.5, label="Foot Strike of Left Foot")
#         line2 = ax.axvline(footoff_ic_time, color="red", linestyle="dotted", linewidth=1.5, label="Toe-off of Left Foot")
#         line3 = ax.axvline(footstrike_opp_time, color="blue", linestyle="solid", linewidth=1.5, label="Foot Strike of Right Foot")
#         line4 = ax.axvline(footoff_opp_time, color="blue", linestyle="dotted", linewidth=1.5, label="Toe-off of Right Foot")

#         ax.set_ylabel(axis_labels[axis])
#         ax.legend()

#     plt.xlabel("Time (s)")
#     plt.suptitle(f"Mean ± SD WBAM - Participant {participant}", fontsize=14)
#     plt.tight_layout()
#     plt.show()


def plot_overall_wbam_mean_sd(wbam_data):
    """
    Plots the overall mean ± SD WBAM data across all participants and trials.
    """
    mean_df = wbam_data.groupby(["Frame", "Phase"])[["X", "Y", "Z"]].mean().reset_index()
    std_df = wbam_data.groupby(["Frame", "Phase"])[["X", "Y", "Z"]].std().reset_index()

    SAMPLING_RATE = 60
    FOOTSTRIKE_IC_FRAME = PREPARATION_FRAMES
    FOOTOFF_OPP_FRAME = PREPARATION_FRAMES + TARGET_SIDESTEP_FRAMES
    footstrike_ic_time = FOOTSTRIKE_IC_FRAME / SAMPLING_RATE
    footoff_opp_time = FOOTOFF_OPP_FRAME / SAMPLING_RATE

    stance_ic_offset = avg_stance_ic_pct / 100 * TARGET_SIDESTEP_FRAMES
    footoff_ic_frame = FOOTSTRIKE_IC_FRAME + stance_ic_offset
    footoff_ic_time = footoff_ic_frame / SAMPLING_RATE

    stance_opp_offset = avg_stance_opp_pct / 100 * TARGET_SIDESTEP_FRAMES
    footstrike_opp_frame = FOOTOFF_OPP_FRAME - stance_opp_offset
    footstrike_opp_time = footstrike_opp_frame / SAMPLING_RATE

    phase_colors = {
        "Preparation": "tab:blue",
        "Sidestep": "tab:orange",
        "Recovery": "tab:green"
    }

    axis_labels = ["Frontal", "Sagittal", "Transverse"]
    fig, axes = plt.subplots(3, 1, figsize=(16, 15), sharex=True)
    legend_handles = []

    for ax, axis, label in zip(axes, ["X", "Y", "Z"], axis_labels):
        for phase, color in phase_colors.items():
            if phase == "Preparation":
                phase_mask = (mean_df["Phase"] == "Preparation") | ((mean_df["Frame"] == FOOTSTRIKE_IC_FRAME))
            elif phase == "Sidestep":
                phase_mask = (mean_df["Phase"] == "Sidestep") | ((mean_df["Frame"] == FOOTOFF_OPP_FRAME))
            else:
                phase_mask = (mean_df["Phase"] == "Recovery")
            line, = ax.plot(mean_df.loc[phase_mask, "Frame"] / SAMPLING_RATE,
                            mean_df.loc[phase_mask, axis],
                            label=f"{phase} phase (mean)", color=color, linewidth=5)
            shade = ax.fill_between(mean_df.loc[phase_mask, "Frame"] / SAMPLING_RATE,
                                    mean_df.loc[phase_mask, axis] - std_df.loc[phase_mask, axis],
                                    mean_df.loc[phase_mask, axis] + std_df.loc[phase_mask, axis],
                                    color=color, alpha=0.2, label=f"{phase} ±1 SD")
            if axis == "X":
                legend_handles.extend([line])

        line1 = ax.axvline(footstrike_ic_time, color="red", linestyle="solid", linewidth=2, label="Left foot strike")
        line2 = ax.axvline(footoff_ic_time, color="red", linestyle="dotted", linewidth=4, label="Left toe-off")
        line3 = ax.axvline(footstrike_opp_time, color="blue", linestyle="solid", linewidth=2, label="Right foot strike")
        line4 = ax.axvline(footoff_opp_time, color="blue", linestyle="dotted", linewidth=4, label="Right toe-off")

        if axis == "X":
            legend_handles.extend([line1, line2, line3, line4])

        ax.axhline(y=0, color="grey", linestyle="--", linewidth=1.5)
        ax.set_ylabel(label, fontsize=38)
        ax.tick_params(axis='both', labelsize=30)
        ax.set_ylim(-0.03, 0.04)

    fig.text(-0.08, 0.5, "Normalized Whole-body Angular Momentum", va='center', rotation='vertical', fontsize=34, fontweight='bold')
    fig.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.0005, 0.5), fontsize=34)
    plt.xlabel("Time (s)", fontsize=38)
    plt.tight_layout(rect=[-0.03, 0, 1, 1])
    fig.savefig("overall_wbam_plot.png", dpi=600, bbox_inches='tight')


plot_overall_wbam_mean_sd(wbam_averaged)


segment_groups = {
    "Trunk (RPV, RTX, RHE)": ["RPV", "RTX", "RHE"],
    "Upper Limbs (Hands, Forearms, Arms)": ["LHA", "RHA", "LFA", "RFA", "LAR", "RAR"],
    "Lower Limbs (Toes to Thighs)": ["LTO", "RTO", "LFT", "RFT", "LSK", "RSK", "LTH", "RTH"]
}

segment_color_groups = {
    "HA": "tab:blue",
    "FA": "tab:red",
    "AR": "tab:green",
    "TO": "tab:blue",
    "FT": "tab:gray",
    "SK": "tab:green",
    "TH": "tab:red",
    "PV": "tab:gray",   # Pelvis
    "TX": "tab:olive",  # Thorax
    "HE": "tab:cyan"    # Head
}

segment_labels = {
    "LHA": "Hand", "RHA": "Hand",
    "LFA": "Forearm", "RFA": "Forearm",
    "LAR": "Arm", "RAR": "Arm",
    "LTO": "Toes", "RTO": "Toes",
    "LFT": "Foot", "RFT": "Foot",
    "LSK": "Shank", "RSK": "Shank",
    "LTH": "Thigh", "RTH": "Thigh",
    "RPV": "Pelvis", "RTX": "Thorax", "RHE": "Head"
}

segment_colors = {
    seg: segment_color_groups.get(seg[1:], "black")
    for seg in segment_labels
}

midline_segments = ["RPV", "RTX", "RHE"]
segment_styles = {
    seg: "-" if seg.startswith("L") or seg in midline_segments else ":"
    for seg in segment_labels
}

subplot_titles = ["Frontal", "Sagittal", "Transverse"]
SAMPLING_RATE = 60
footstrike_ic_time = PREPARATION_FRAMES / SAMPLING_RATE
footoff_opp_time = (PREPARATION_FRAMES + TARGET_SIDESTEP_FRAMES) / SAMPLING_RATE


import scipy.stats as stats
from scipy.stats import shapiro
import pingouin as pg

sidestep_df = wbam_averaged[wbam_averaged["Phase"] == "Sidestep"].copy()

peak_lwb = (
    sidestep_df.groupby(["Participant"])[["X", "Y", "Z"]]
    .agg(lambda x: x.max() - x.min())
    .reset_index()
)

peak_lwb.rename(columns={"X": "Frontal", "Y": "Sagittal", "Z": "Transverse"}, inplace=True)

df_long = peak_lwb.melt(id_vars=["Participant"], 
                         value_vars=["Frontal", "Sagittal", "Transverse"],
                         var_name="Plane", 
                         value_name="Peak_LWB")

print(df_long.groupby("Plane")["Peak_LWB"].describe())

# Normality of residuals and Levene’s Test for Homogeneity of Variance
for plane in df_long["Plane"].unique():
    stat, p = shapiro(df_long[df_long["Plane"] == plane]["Peak_LWB"])
    print(f"Shapiro-Wilk test for {plane}: W = {stat:.4f}, p = {p:.4f}")

levene_stat, levene_p = stats.levene(
    df_long[df_long["Plane"] == "Frontal"]["Peak_LWB"],
    df_long[df_long["Plane"] == "Sagittal"]["Peak_LWB"],
    df_long[df_long["Plane"] == "Transverse"]["Peak_LWB"]
)

print(f"Levene’s test: W = {levene_stat:.4f}, p = {levene_p:.4f}")

# Choose ANOVA type based on variance equality
if levene_p > 0.05:
    print("\n Variances are equal: Running One-Way ANOVA...")
    anova_result = stats.f_oneway(
        df_long[df_long["Plane"] == "Frontal"]["Peak_LWB"],
        df_long[df_long["Plane"] == "Sagittal"]["Peak_LWB"],
        df_long[df_long["Plane"] == "Transverse"]["Peak_LWB"]
    )
    print(f"One-Way ANOVA: F = {anova_result.statistic:.4f}, p = {anova_result.pvalue:.4f}")
else:
    print("\n Variances are NOT equal: Running Welch ANOVA instead...")
    welch_result = pg.welch_anova(dv="Peak_LWB", between="Plane", data=df_long)
    print(welch_result)

# Post-Hoc: Games-Howell
gh_results = pg.pairwise_gameshowell(dv='Peak_LWB', between='Plane', data=df_long)
print(gh_results)


# PCA for segment contributions
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

planes = {
    "Frontal": [f"X_{seg}" for seg in ["LHA", "LFA", "LAR", "RHA", "RFA", "RAR",
                                       "LTO", "LFT", "LSK", "LTH", "RTO", "RFT", "RSK", "RTH",
                                       "RPV", "RTX", "RHE"]],
    "Sagittal": [f"Y_{seg}" for seg in ["LHA", "LFA", "LAR", "RHA", "RFA", "RAR",
                                        "LTO", "LFT", "LSK", "LTH", "RTO", "RFT", "RSK", "RTH",
                                        "RPV", "RTX", "RHE"]],
    "Transverse": [f"Z_{seg}" for seg in ["LHA", "LFA", "LAR", "RHA", "RFA", "RAR",
                                          "LTO", "LFT", "LSK", "LTH", "RTO", "RFT", "RSK", "RTH",
                                          "RPV", "RTX", "RHE"]],
}

sidestep_segments = segment_averaged[segment_averaged["Phase"] == "Sidestep"]

pca_data = sidestep_segments.pivot_table(
    index=["Frame"], 
    columns="Segment",
    values=["X", "Y", "Z"]
)

pca_data.columns = [f"{axis}_{seg}" for axis, seg in pca_data.columns]
pca_data.reset_index(drop=True, inplace=True)
pca_results = {}

# Run PCA separately for each plane
for plane_name, segments in planes.items():
    print(f"\n Running PCA for {plane_name} Plane")

    pca_input_plane = pca_data[segments]
    scaler = StandardScaler()
    pca_input_standardized = scaler.fit_transform(pca_input_plane)

    pca_full = PCA()
    pca_full.fit(pca_input_standardized)

    explained_variance = pca_full.explained_variance_ratio_ * 100  # Convert to percentage
    eigenvalues = pca_full.explained_variance_  # Get eigenvalues

    # Kaiser's Criterion (Eigenvalues > 1)
    num_pcs_kaiser = np.sum(eigenvalues > 1)
    print(f"Number of PCs selected using Kaiser's Criterion (Eigenvalues > 1): {num_pcs_kaiser}")

    # Run PCA with the selected number of PCs
    pca = PCA(n_components=num_pcs_kaiser)
    principal_components = pca.fit_transform(pca_input_standardized)

    explained_variance_selected = pca.explained_variance_ratio_ * 100
    print(f"Variance Explained by {num_pcs_kaiser} PCs: {sum(explained_variance_selected):.2f}%")
    
    # Extract PCA Loadings
    loadings = pd.DataFrame(pca.components_.T, index=segments, columns=[f"PC{i+1}" for i in range(num_pcs_kaiser)])
    
    # Store Tuning Coefficients
    tuning_df = pd.DataFrame(principal_components, columns=[f"PC{i+1}" for i in range(num_pcs_kaiser)])
    tuning_df["Frame"] = range(len(principal_components))

    pca_results[plane_name] = {
        "explained_variance": explained_variance_selected,
        "num_pcs": num_pcs_kaiser,
        "loadings": loadings,
        "tuning_coefficients": tuning_df
    }




import matplotlib.patches as mpatches
import matplotlib.lines as mlines

segment_full_names = ["LHA", "LFA", "LAR", "RHA", "RFA", "RAR",
                      "LTO", "LFT", "LSK", "LTH", "RTO", "RFT", "RSK", "RTH",
                      "RPV", "RTX", "RHE"]

segment_display_labels = [seg.replace("RPV", "PV").replace("RTX", "TX").replace("RHE", "HE")
                          for seg in segment_full_names]

segment_label_legend = {
    "LHA": "Left hand", "LFA": "Left forearm", "LAR": "Left upper arm",
    "RHA": "Right hand", "RFA": "Right forearm", "RAR": "Right upper arm",
    "LTO": "Left toes", "LFT": "Left foot", "LSK": "Left shank", "LTH": "Left thigh",
    "RTO": "Right toes", "RFT": "Right foot", "RSK": "Right shank", "RTH": "Right thigh",
    "RPV": "Pelvis", "RTX": "Thorax", "RHE": "Head"
}

segment_colors = {
    "LHA": "#E5A84C", "LFA": "#E5A84C", "LAR": "#E5A84C",
    "LTO": "#72A7E0", "LFT": "#72A7E0", "LSK": "#72A7E0", "LTH": "#72A7E0",
    "RHA": "#E5A84C", "RFA": "#E5A84C", "RAR": "#E5A84C",
    "RTO": "#72A7E0", "RFT": "#72A7E0", "RSK": "#72A7E0", "RTH": "#72A7E0",
    "RPV": "#74B97E", "RTX": "#74B97E", "RHE": "#74B97E"
}

segment_hatches = {
    "LHA": "//", "LFA": "//", "LAR": "//", "LTO": "//", "LFT": "//", "LSK": "//", "LTH": "//",
    "RHA": "\\", "RFA": "\\", "RAR": "\\", "RTO": "\\", "RFT": "\\", "RSK": "\\", "RTH": "\\",
    "RPV": "", "RTX": "", "RHE": ""
}

legend_display = {
    seg.replace("RPV", "PV").replace("RTX", "TX").replace("RHE", "HE"): name
    for seg, name in segment_label_legend.items()
}

for plane_name in planes.keys():
    loadings_df = pca_results[plane_name]["loadings"]
    num_pcs = pca_results[plane_name]["num_pcs"]

    fig, axes = plt.subplots(num_pcs, 1, figsize=(18, 4 * num_pcs), sharex=True)
    if num_pcs == 1:
        axes = [axes]

    for pc_idx in range(num_pcs):
        ax = axes[pc_idx]
        pc_values = loadings_df.iloc[:, pc_idx]
        x = range(len(segment_full_names))

        for i, seg in enumerate(segment_full_names):
            ax.bar(x[i], pc_values.iloc[i],
                   color=segment_colors[seg],
                   hatch=segment_hatches[seg],
                   edgecolor='black',
                   alpha=0.9)

        ax.axhline(0, color='black', linewidth=1)
        ax.set_ylabel(f"PC{pc_idx+1}\nLoading Value", fontsize=28, labelpad=10)
        variance = pca_results[plane_name]["explained_variance"][pc_idx]
        ax.set_title(f"{plane_name} Plane - PC{pc_idx+1} ({variance:.1f}%)", fontsize=30)
        ax.set_xticks(x)
        ax.set_xticklabels([legend_display[x] for x in segment_display_labels], fontsize=20, ha="right", rotation=45)
        ax.set_yticks([-0.2, 0.0, 0.2, 0.4])
        ax.tick_params(axis='y', labelsize=26)
        ax.grid(True, linestyle="--", alpha=0.6)

    if len(axes) > num_pcs:
        for i in range(num_pcs, len(axes)):
            fig.delaxes(axes[i])


    color_patches = [
        mpatches.Patch(color='#E5A84C', label='Upper limbs'),
        mpatches.Patch(color='#72A7E0', label='Lower limbs'),
        mpatches.Patch(color='#74B97E', label='Trunk'),
        mpatches.Patch(facecolor='white', edgecolor='black', hatch='//', label='Left side'),
        mpatches.Patch(facecolor='white', edgecolor='black', hatch='\\', label='Right side'),
        mpatches.Patch(facecolor='white', edgecolor='black', hatch='', label='Core')
    ]


    legend1 = fig.legend(handles=color_patches, ncols=2, bbox_to_anchor=(0.5, -0.1), fontsize=20, loc="lower center", frameon=True)


    plt.subplots_adjust(bottom=0.4)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    filename = f"pca_loadings_{plane_name.lower()}.png"
    fig.savefig(filename, dpi=600, bbox_inches='tight')
    plt.show()


plt.close()

# Violin plot for peak difference of WBAM in three planes
import seaborn as sns
import matplotlib.pyplot as plt

plane_colors = {
    "Frontal": "tab:blue",
    "Sagittal": "tab:orange",
    "Transverse": "tab:green"
}

fig, ax = plt.subplots(figsize=(10, 6))

for pp in df_long["Participant"].unique():
    y_vals = [
        df_long.loc[(df_long["Participant"]==pp) & (df_long["Plane"]=="Frontal"), "Peak_LWB"].iloc[0],
        df_long.loc[(df_long["Participant"]==pp) & (df_long["Plane"]=="Sagittal"), "Peak_LWB"].iloc[0],
        df_long.loc[(df_long["Participant"]==pp) & (df_long["Plane"]=="Transverse"), "Peak_LWB"].iloc[0]
    ]
    y_vals = [float(y) for y in y_vals]
    x_vals = [0, 1, 2]
    ax.plot(x_vals, y_vals, color=(0.3, 0.3, 0.3), zorder=2, alpha=0.1)

sns.violinplot(
    x="Plane",
    y="Peak_LWB",
    data=df_long,
    palette=plane_colors,
    inner="quart",
    linewidth=1.5,
    ax=ax,
)

sns.stripplot(
    x="Plane",
    y="Peak_LWB",
    data=df_long,
    color="black",
    size=5,
    jitter=False,       
    dodge=False,
    alpha=0.7,
    ax=ax
)

ax.set_xlabel("Anatomical Planes", fontsize=20)
ax.set_ylabel("Peak Differences in\nWhole-body Angular Momentum", fontsize=16)
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=18)

fig.tight_layout()
fig.savefig("violin_plot_peak_difference_with_points.png", dpi=300, bbox_inches="tight")


plt.close()