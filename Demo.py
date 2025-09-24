import os
import torch
import h5py
import torch.nn as nn
import torchkbnufft as tkbn
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from time import time
from argparse import ArgumentParser
from scipy.signal import find_peaks, savgol_filter
from scipy.linalg import svd
from torch.utils.data import TensorDataset, DataLoader
from functions.Toolbox import crop, normbg, MCNUFFT
from functions.ULS_net import ULS_net


def get_config():
    parser = ArgumentParser(description="GRACE_MORE")
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--wins", type=int, default=3)
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--checkpoint", type=str, default="model")
    parser.add_argument("--output_dir", type=str, default="results")
    return parser.parse_args()


config = get_config()
layers = config.layers
wins = config.wins
gpus = config.gpus

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with h5py.File("./data/liver/rawdata.mat", "r") as f:
    raw_data = f["rawdata"][:]

sensitivity_mat = sio.loadmat("./data/liver/smap.mat")
traj_mat = sio.loadmat("./data/liver/traj.mat")
freq_mat = sio.loadmat("./data/liver/freq.mat")
full_mat = sio.loadmat("./data/liver/full_sampling.mat")
mask_mat = sio.loadmat("./data/liver/Mask.mat")

sensitivity_map = np.transpose(sensitivity_mat["smap"])
trajectory = np.transpose(traj_mat["traj"])
freq = np.transpose(freq_mat["freqs"])
full = np.transpose(full_mat["img_recon"])
mask = np.squeeze(mask_mat["Mask"]).astype(bool)
mask0 = np.squeeze(mask_mat["Mask0"]).astype(bool)
mask1 = np.squeeze(mask_mat["Mask1"]).astype(bool)

raw_data = np.expand_dims(raw_data, axis=0)
sensitivity_map = np.expand_dims(sensitivity_map, axis=0)

ref_img = np.transpose(raw_data, [0, 4, 3, 1, 2])
coil_maps = np.transpose(sensitivity_map, [0, 2, 4, 3, 1])
coil_maps = coil_maps.astype("complex128")
full = np.transpose(full, (1, 2, 0))

num_items = ref_img.shape[0]
num_samples = ref_img.shape[1]
num_frames = ref_img.shape[3]
num_coils = coil_maps.shape[1]

frames_per_group = wins
group_idx = np.arange(num_frames, dtype=int)
group_idx = np.lib.stride_tricks.sliding_window_view(group_idx, (frames_per_group,))
group_idx = torch.tensor(group_idx)

real_part = ref_img["real"].astype(np.float32)
imag_part = ref_img["imag"].astype(np.float32)
ref_complex = (ref_img["real"] + 1j * ref_img["imag"]).astype(np.complex64)

ref_tensor = torch.from_numpy(ref_complex)
coil_tensor = torch.tensor(coil_maps, dtype=torch.complex64)

test_dataset = TensorDataset(ref_tensor, coil_tensor)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

ULS_net = ULS_net(layers)
ULS_net = nn.DataParallel(ULS_net, device_ids=[0])
ULS_net = ULS_net.to(device)

model_save_path = f"./{config.checkpoint}/ULS_net"
ULS_net.load_state_dict(
    torch.load(f"{model_save_path}/net_params.pkl", map_location="cuda:0")
)

recon_name = f"./{config.output_dir}/liver_GRACE_MORE.mat"
resp_name = f"./{config.output_dir}/liver_GRACE_MORE_Resp.mat"

output_dir = os.path.join(config.output_dir)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def initialize_nufft(sample_pts, group_size, traj_U):
    os_factor = 2
    img_dims = (sample_pts // os_factor, sample_pts // os_factor)
    grid_dims = (sample_pts, sample_pts)

    traj_U = np.transpose(traj_U, [2, 0, 1])
    num_spokes = traj_U.shape[1]
    kspace_traj = traj_U * 2 * np.pi
    a, m, n = kspace_traj.shape
    kspace_traj = kspace_traj.reshape(a * m, n).flatten()
    kspace_traj = np.stack((np.real(kspace_traj), np.imag(kspace_traj)), axis=0)
    kspace_traj = torch.tensor(kspace_traj, dtype=torch.float)

    density_comp = tkbn.calc_density_compensation_function(
        ktraj=kspace_traj, im_size=img_dims
    ).squeeze()

    traj_groups = np.zeros((2, num_spokes * sample_pts, group_size), dtype=float)
    comp_groups = np.zeros((num_spokes * sample_pts, group_size), dtype=complex)

    for idx in range(group_size):
        start = idx * num_spokes * sample_pts
        end = (idx + 1) * num_spokes * sample_pts
        traj_groups[:, :, idx] = kspace_traj[:, start:end]
        comp_groups[:, idx] = density_comp[start:end]

    traj_groups = torch.tensor(traj_groups, dtype=torch.float)
    comp_groups = torch.tensor(comp_groups, dtype=torch.complex64)

    forward_nufft = tkbn.KbNufft(im_size=img_dims, grid_size=grid_dims)
    adjoint_nufft = tkbn.KbNufftAdjoint(im_size=img_dims, grid_size=grid_dims)
    return traj_groups, comp_groups, forward_nufft, adjoint_nufft


def perform_radial_downsampling(kspace_data, nufft_operator):
    undersampled_img = nufft_operator(inv=True, data=kspace_data)
    undersampled_img = undersampled_img.to(kspace_data.device)
    return undersampled_img


def AP_PCA(kspace_data):
    num_samples_, num_spokes, num_coils_ = kspace_data.shape
    center_amplitude = np.abs(kspace_data[num_samples_ // 2, :, :])
    center_phase = np.angle(kspace_data[num_samples_ // 2, :, :])

    center_amplitude = np.expand_dims(center_amplitude, axis=0)
    center_phase = np.expand_dims(center_phase, axis=0)
    combined_data = np.concatenate((center_amplitude, center_phase), axis=0)

    principal_components = np.zeros((num_spokes, 2 * num_coils_))
    component_idx = 0

    for coil_idx in range(num_coils_):
        coil_data = np.transpose(combined_data[:, :, coil_idx], (1, 0))
        coil_data = np.abs(coil_data.reshape((num_spokes, -1)))
        covariance_matrix = np.cov(coil_data, rowvar=False)
        eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
        sorted_indices = np.argsort(-eigen_values)
        eigen_vectors = eigen_vectors[:, sorted_indices]
        pca_projection = np.dot(eigen_vectors.T, coil_data.T).T

        for comp in range(2):
            filtered_signal = savgol_filter(pca_projection[:, comp], 10, 1)
            filtered_signal = filtered_signal - np.min(filtered_signal)
            filtered_signal = filtered_signal / np.max(filtered_signal)
            principal_components[:, component_idx] = filtered_signal
            component_idx += 1

    threshold = 0.93
    respiration_signal, _ = coil_clustering(principal_components, threshold)
    respiration_signal = respiration_signal - np.min(respiration_signal)
    respiration_signal = respiration_signal / np.max(respiration_signal)
    return respiration_signal


def coil_clustering(data, threshold):
    correlation_matrix = np.corrcoef(data.T)
    mask_ = np.zeros_like(correlation_matrix)
    mask_[np.abs(correlation_matrix) > threshold] = 1

    U, _, _ = svd(mask_, full_matrices=False)
    first_singular_vector = np.abs(U[:, 0])

    coil_subgroup = np.zeros(len(first_singular_vector))
    coil_subgroup[first_singular_vector > 0.1] = 1

    aggregated_signal = np.zeros(data.shape[0])
    coil_ids = coil_subgroup.copy()

    if np.sum(coil_subgroup) < 2:
        print("Clustering failed, only one coil selected...")

    subgroup_indices = np.where(coil_subgroup == 1)[0]
    for coil_idx in range(data.shape[1]):
        if coil_subgroup[coil_idx] > 0:
            if correlation_matrix[subgroup_indices[0], coil_idx] > 0:
                aggregated_signal += data[:, coil_idx]
            else:
                aggregated_signal -= data[:, coil_idx]
                coil_ids[coil_idx] = -coil_ids[coil_idx]

    aggregated_signal /= np.sum(coil_subgroup)
    return aggregated_signal, coil_ids


def filter_signal(signal):
    fft_result = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(signal))

    shifted_fft = np.fft.fftshift(fft_result)
    shifted_frequencies = np.fft.fftshift(frequencies)

    frequency_mask = (shifted_frequencies >= -0.5) & (shifted_frequencies <= 0.5)
    filtered_fft = np.zeros_like(shifted_fft)
    filtered_fft[frequency_mask] = shifted_fft[frequency_mask]

    filtered_signal = np.fft.ifft(np.fft.ifftshift(filtered_fft))
    filtered_signal = np.real(filtered_signal)
    filtered_signal = savgol_filter(filtered_signal, window_length=3, polyorder=1)
    return filtered_signal


oversample_factor = 2
Ny = int(num_samples / oversample_factor)
Nx = int(num_samples / oversample_factor)
mC = 0

Zplot = np.zeros((Ny, Nx, num_frames), dtype=complex)
undersample = np.zeros((Ny, Nx, num_frames, num_items), dtype=complex)

print()
print("GRACE_MORE Reconstruction Start")

with torch.no_grad():
    for idx in range(num_items):
        x_test = ref_tensor[idx, :, :, :, :]
        coil_map_sample = coil_tensor[idx, :, :, :, :]
        coil_map_sample = coil_map_sample.to(device=device, dtype=torch.complex64)

        for p in range(num_frames):
            if p == 100:
                rawdata_group = x_test[:, :, group_idx[p - 1, :], :]
                m_range = [1, 0, 2]
            elif p == 101:
                rawdata_group = x_test[:, :, group_idx[p - 2, :], :]
                m_range = [2, 0, 1]
            else:
                rawdata_group = x_test[:, :, group_idx[p, :], :]
                m_range = [0, 1, 2]

            nline_calculated = False

            for m in m_range:
                Signal = AP_PCA(rawdata_group[:, :, m, :])
                raw_temp = rawdata_group[:, :, m, :]
                traj_temp = trajectory[:, :]

                if not nline_calculated:
                    filtered_signal = filter_signal(Signal)
                    normalized_main_freqs = freq[300 * p : 300 * (p + 1), :]

                    pause_threshold = 0.7
                    pause_indices = np.where(normalized_main_freqs < pause_threshold)[0]

                    if 50 < p < 60:
                        nline = 270
                    else:
                        pks, properties = find_peaks(
                            filtered_signal, height=0.09, distance=200
                        )
                        threshold_val = 0.5
                        signal_below_thresh = filtered_signal < threshold_val
                        pause_signals = np.isin(
                            np.arange(len(filtered_signal)), pause_indices
                        ) & signal_below_thresh
                        indices = np.where(pause_signals)[0]
                        sorted_indices = np.argsort(filtered_signal[indices])
                        sorted_indices = indices[sorted_indices]

                        for pk in pks:
                            if pk in indices:
                                start = max(pk - 50, 0)
                                end = min(pk + 50, len(filtered_signal) - 1)
                                to_remove = np.arange(start, end + 1)
                                sorted_indices = np.setdiff1d(sorted_indices, to_remove)
                        nline = len(sorted_indices)

                    if nline < 70:
                        nline = 200

                    traj_U = np.zeros((nline, 256, 3), dtype=complex)
                    rawdata_U = np.zeros((256, nline, 48, 3), dtype=complex)
                    nline_calculated = True

                index_order = np.argsort(Signal)
                rawdata_temp = raw_temp[:, index_order, :]
                traj_temp = traj_temp[index_order, :]
                traj_U[:, :, m] = traj_temp[:nline, :]
                rawdata_U[:, :, :, m] = rawdata_temp[:, :nline, :]

            ktraj, dcomp, nufft_ob, adjnufft_ob = initialize_nufft(
                num_samples, frames_per_group, traj_U
            )
            ktraj = ktraj.to(device)
            dcomp = dcomp.to(device)
            nufft_ob = nufft_ob.to(device)
            adjnufft_ob = adjnufft_ob.to(device)
            smap0 = coil_map_sample[:, :, :, p]
            param_E = MCNUFFT(nufft_ob, adjnufft_ob, ktraj, dcomp, smap0)

            rawdata_U = np.transpose(rawdata_U, [0, 1, 3, 2]) * 1e-7
            gnd_it = rawdata_U
            gnd = torch.tensor(gnd_it, dtype=torch.complex64).to(device)
            gnd = torch.transpose(gnd, 2, 3)
            gnd = torch.transpose(gnd, 0, 2)

            a, m_dim, n_dim, b = gnd.size()
            gnd = gnd.reshape(a, m_dim * n_dim, b)

            M0 = perform_radial_downsampling(gnd, param_E)

            torch.cuda.synchronize()
            start_time = time()
            L, S, loss_L, loss_S = ULS_net(M0, param_E, gnd)
            M_recon_tensor = L + S
            M_recon = np.squeeze(M_recon_tensor.cpu().data.numpy())
            torch.cuda.synchronize()
            end_time = time()
            elapsed_time = end_time - start_time

            X_rec = crop(
                normbg(M_recon, mask),
                int(num_samples / oversample_factor) - mC,
                int(num_samples / oversample_factor) - mC,
            )

            if p == 100:
                Zplot[:, :, p] = X_rec[:, :, 1]
            elif p == 101:
                Zplot[:, :, p] = X_rec[:, :, 2]
            else:
                Zplot[:, :, p] = X_rec[:, :, 0]

        undersample[:, :, :, idx] = Zplot

slice_idx = 21
rec_ = abs(Zplot[..., slice_idx].T)
full_ = abs(np.flipud(full[..., slice_idx].T))

SI_liver_full = full_[mask0].mean()
SI_liver_rec = rec_[mask0].mean()
eps = 1e-12
SD_bg_full = np.std(full_[mask1], ddof=1)
SD_bg_rec = np.std(rec_[mask1], ddof=1)
snr_full = SI_liver_full / max(SD_bg_full, eps)
snr_rec = SI_liver_rec / max(SD_bg_rec, eps)
print(f"SNR(full) = {snr_full:.3f} | SNR(rec) = {snr_rec:.3f}")

rec_vis = rec_
full_vis = full_
vis = np.concatenate([full_vis, rec_vis], axis=1)

plt.figure()
plt.subplot(1, 1, 1)
plt.imshow(vis, cmap="gray", vmin=0, vmax=0.3)
plt.axis("off")
plt.tight_layout()
plt.show()

sio.savemat(recon_name, {"undersample": undersample})

print()
print("GRACE_MORE Reconstruction end.")
