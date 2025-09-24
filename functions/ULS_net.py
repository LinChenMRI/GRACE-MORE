import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from functions.LS import Project, Wxs, Wtxs
from functions.Unet_ECA import UNet3D, ECAAttention

dtype = torch.complex64


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lambda_L = nn.Parameter(torch.tensor(0.01))
        self.lambda_S = nn.Parameter(torch.tensor(0.01))
        self.lambda_spatial_L = nn.Parameter(torch.tensor(5e-2))
        self.lambda_spatial_S = nn.Parameter(torch.tensor(5e-2))
        self.gamma = nn.Parameter(torch.tensor(0.5))
        self.lambda_step = nn.Parameter(torch.tensor(0.1))

        self.conv1_backward_l = nn.Parameter(
            init.xavier_normal_(torch.Tensor(32, 32, 3, 3, 3))
        )
        self.conv2_backward_l = nn.Parameter(
            init.xavier_normal_(torch.Tensor(32, 32, 3, 3, 3))
        )
        self.conv3_backward_l = nn.Parameter(
            init.xavier_normal_(torch.Tensor(1, 32, 3, 3, 3))
        )

        self.conv1_forward_s = nn.Parameter(
            init.xavier_normal_(torch.Tensor(32, 1, 3, 3, 3))
        )
        self.conv2_forward_s = nn.Parameter(
            init.xavier_normal_(torch.Tensor(32, 32, 3, 3, 3))
        )
        self.conv3_forward_s = nn.Parameter(
            init.xavier_normal_(torch.Tensor(32, 32, 3, 3, 3))
        )

        self.conv1_backward_s = nn.Parameter(
            init.xavier_normal_(torch.Tensor(32, 32, 3, 3, 3))
        )
        self.conv2_backward_s = nn.Parameter(
            init.xavier_normal_(torch.Tensor(32, 32, 3, 3, 3))
        )
        self.conv3_backward_s = nn.Parameter(
            init.xavier_normal_(torch.Tensor(1, 32, 3, 3, 3))
        )

        self.unet_model = UNet3D(in_channels=in_channels, num_classes=out_channels)
        self.eca1 = ECAAttention(32)
        self.eca2 = ECAAttention(32)

    def forward(
        self, M0, param_E, param_d, L, S, pt_L, pt_S, p_L, p_S
    ):
        c = self.lambda_step / self.gamma
        nx, ny, nt = M0.size()
        device = param_d.device

        temp_data = (L + S).view(nx, ny, nt)
        temp_data = param_E(inv=False, data=temp_data).to(device)
        gradient = param_E(inv=True, data=temp_data- param_d).view(nx * ny, nt).to(
            device
        )

        pb_L = F.conv3d(p_L, self.conv1_backward_l, padding=1)
        pb_L = F.relu(pb_L)
        pb_L = F.conv3d(pb_L, self.conv2_backward_l, padding=1)
        pb_L = F.relu(pb_L)
        pb_L = F.conv3d(pb_L, self.conv3_backward_l, padding=1)
        pb_L = torch.squeeze(pb_L).view(2, nx * ny, nt)
        pb_L = pb_L[0] + 1j * pb_L[1]

        y_L = L - self.gamma * (gradient + pt_L + pb_L)

        U, S_vals, Vh = torch.linalg.svd(c * y_L + pt_L, full_matrices=False)
        temp_S = torch.diag(Project(S_vals, self.lambda_L))
        pt_L = U.mm(temp_S).mm(Vh)

        temp_y_L_input = torch.cat((torch.real(y_L), torch.imag(y_L)), 0).to(
            torch.float32
        )
        temp_y_L_input = temp_y_L_input.view(2, nx, ny, nt).unsqueeze(1)
        temp_y_L_output = self.unet_model(temp_y_L_input)
        temp_y_L = temp_y_L_output + p_L
        temp_y_L = temp_y_L[0] + 1j * temp_y_L[1]
        p_L = Project(c * temp_y_L, self.lambda_spatial_L)

        p_L = torch.cat((torch.real(p_L), torch.imag(p_L)), 0).to(torch.float32)
        p_L = p_L.view(2, 32, nx, ny, nt)
        pb_L = F.conv3d(p_L, self.conv1_backward_l, padding=1)
        pb_L = F.relu(pb_L)
        pb_L = F.conv3d(pb_L, self.conv2_backward_l, padding=1)
        pb_L = F.relu(pb_L)
        pb_L_output = F.conv3d(pb_L, self.conv3_backward_l, padding=1)
        pb_L = pb_L_output.view(2, nx * ny, nt)
        pb_L = pb_L[0] + 1j * pb_L[1]

        L = L - self.gamma * (gradient + pt_L + pb_L)
        adjloss_L = temp_y_L_output * p_L - pb_L_output * temp_y_L_input

        pb_S = F.conv3d(p_S, self.conv1_backward_s, padding=1)
        pb_S = F.relu(pb_S)
        pb_S = F.conv3d(pb_S, self.conv2_backward_s, padding=1)
        pb_S = F.relu(pb_S)
        pb_S = F.conv3d(pb_S, self.conv3_backward_s, padding=1)
        pb_S = pb_S.view(2, nx * ny, nt)
        pb_S = pb_S[0] + 1j * pb_S[1]

        y_S = S - self.gamma * (gradient + Wtxs(pt_S) + pb_S)
        pt_S = Project(c * Wxs(y_S) + pt_S, self.lambda_S)

        temp_y_S_input = torch.cat((torch.real(y_S), torch.imag(y_S)), 0).to(
            torch.float32
        )
        temp_y_S_input = temp_y_S_input.view(2, nx, ny, nt).unsqueeze(1)
        temp_y_S = F.conv3d(temp_y_S_input, self.conv1_forward_s, padding=1)
        temp_y_S = F.relu(temp_y_S)
        temp_y_S = self.eca1(temp_y_S)
        temp_y_S = F.conv3d(temp_y_S, self.conv2_forward_s, padding=1)
        temp_y_S = F.relu(temp_y_S)
        temp_y_S = self.eca2(temp_y_S)
        temp_y_S_output = F.conv3d(temp_y_S, self.conv3_forward_s, padding=1)

        temp_y_Sp = temp_y_S_output + p_S
        temp_y_Sp = temp_y_Sp[0] + 1j * temp_y_Sp[1]
        p_S = Project(c * temp_y_Sp, self.lambda_spatial_S)

        p_S = torch.cat((torch.real(p_S), torch.imag(p_S)), 0).to(torch.float32)
        p_S = p_S.view(2, 32, nx, ny, nt)
        pb_S = F.conv3d(p_S, self.conv1_backward_s, padding=1)
        pb_S = F.relu(pb_S)
        pb_S = F.conv3d(pb_S, self.conv2_backward_s, padding=1)
        pb_S = F.relu(pb_S)
        pb_S_output = F.conv3d(pb_S, self.conv3_backward_s, padding=1)
        pb_S = pb_S_output.view(2, nx * ny, nt)
        pb_S = pb_S[0] + 1j * pb_S[1]

        S = S - self.gamma * (gradient + Wtxs(pt_S) + pb_S)
        adjloss_S = temp_y_S_output * p_S - pb_S_output * temp_y_S_input

        return [L, S, adjloss_L, adjloss_S, pt_L, pt_S, p_L, p_S]


class ULS_net(nn.Module):
    def __init__(self, layers, in_channels=1, out_channels=32, dtype=torch.complex64):
        super().__init__()
        self.layers = layers
        self.fcs = nn.ModuleList(
            [
                BasicBlock(in_channels=in_channels, out_channels=out_channels)
                for _ in range(layers)
            ]
        )
        self.dtype = dtype

    def forward(self, M0, param_E, param_d):
        M0 = M0[..., 0] + 1j * M0[..., 1]
        param_d = param_d[..., 0] + 1j * param_d[..., 1]

        nx, ny, nt = M0.shape
        device = param_d.device

        L = torch.zeros((nx * ny, nt), dtype=self.dtype, device=device)
        S = torch.zeros((nx * ny, nt), dtype=self.dtype, device=device)
        pt_L = torch.zeros((nx * ny, nt), dtype=self.dtype, device=device)
        pt_S = torch.zeros((nx * ny, nt), dtype=self.dtype, device=device)
        p_L = torch.zeros((2, 32, nx, ny, nt), dtype=torch.float32, device=device)
        p_S = torch.zeros((2, 32, nx, ny, nt), dtype=torch.float32, device=device)

        layers_adj_L = []
        layers_adj_S = []

        for fc in self.fcs:
            L, S, layer_adj_L, layer_adj_S, pt_L, pt_S, p_L, p_S = fc(
                M0, param_E, param_d, L, S, pt_L, pt_S, p_L, p_S
            )
            layers_adj_L.append(layer_adj_L)
            layers_adj_S.append(layer_adj_S)

        L = L.view(nx, ny, nt)
        S = S.view(nx, ny, nt)
        return L, S, layers_adj_L, layers_adj_S
