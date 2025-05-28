import numpy as np
import torch
import torch.nn.functional as F
import cv2
from ...geometry.wrappers import Camera, Pose


def build_K_matrix(f, c):
    """
    Construct batched camera intrinsic matrices from fx, fy and cx, cy.

    Args:
        f: [B, 2] focal lengths (fx, fy)
        c: [B, 2] principal points (cx, cy)

    Returns:
        K: [B, 3, 3] camera intrinsic matrices
    """
    B = f.shape[0]
    K = torch.zeros(B, 3, 3, device=f.device, dtype=f.dtype)
    K[:, 0, 0] = f[:, 0]  # fx
    K[:, 1, 1] = f[:, 1]  # fy
    K[:, 0, 2] = c[:, 0]  # cx
    K[:, 1, 2] = c[:, 1]  # cy
    K[:, 2, 2] = 1.0
    return K


def backproject_keypoints(kpts2d, depth, K):
    """
    kpts2d: [N, 2] tensor
    depth: [H, W] tensor
    K: [3, 3] tensor
    Returns: [N, 3] 3D points in camera coordinates
    """
    H, W = depth.shape
    u = kpts2d[:, 0].long().clamp(0, W - 1)
    v = kpts2d[:, 1].long().clamp(0, H - 1)
    z = depth[v, u]
    valid = z > 0

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x = (kpts2d[:, 0] - cx) * z / fx
    y = (kpts2d[:, 1] - cy) * z / fy
    pts3d = torch.stack([x, y, z], dim=-1)
    return pts3d[valid], kpts2d[valid], valid


def solve_pnp_ransac(kpts0, kpts1, matches0, cam0, cam1, depth0):
    """
    Estimate T_0to1 using PnP with depth0 and 2D-2D matches.
    Inputs:
        kpts0, kpts1: [N, 2] 2D keypoints
        matches0: [N] matched indices in kpts1 or -1
        cam0, cam1: camera objects with intrinsics f, c
        depth0: [H, W] depth map for view0
    Returns:
        T_0to1: [4, 4] torch SE(3) matrix
        0 is follower, 1 is leader
        T_0to1 = T_0^1 = Follower robot's pose w.r.t leader robot
    """
    K0_batch = build_K_matrix(cam0.f, cam0.c)  # for backprojection
    K1_batch = build_K_matrix(cam1.f, cam1.c)  # for projection
    B = kpts0.shape[0]
    T_list = []
    identity_pose = Pose.from_Rt(
                torch.eye(3, device=kpts0.device),
                torch.zeros(3, device=kpts0.device)
            )
    
    for b in range(B):
        m0 = matches0[b]  # [N]
        valid = m0 > -1
        if valid.sum() < 6:
            T_list.append(identity_pose)
            continue

        idx0 = torch.arange(kpts0.shape[1], device=kpts0.device)[valid]
        idx1 = m0[valid]

        pts0 = kpts0[b][idx0]
        pts1 = kpts1[b][idx1]

        pts3d, valid_pts0, valid = backproject_keypoints(pts0, depth0[b], K0_batch[b])
        pts2d = pts1[valid]   # filter with valid z


        if len(pts2d) < 6: # TODO: need another back up plan than identity
            T_list.append(identity_pose)
            continue

        # Convert to NumPy
        pts3d_np = pts3d.cpu().numpy().astype(np.float32)
        pts2d_np = pts2d.cpu().numpy().astype(np.float32)
        K_np = K1_batch[b].cpu().numpy().astype(np.float32) #K1 leader

        # Solve PnP
        success, rvec, tvec, _ = cv2.solvePnPRansac(pts3d_np, pts2d_np, K_np, None)
        if not success: # TODO: need another back up plan than identity or do None
            T_list.append(identity_pose)
            continue

        R, _ = cv2.Rodrigues(rvec)
        R_torch = torch.from_numpy(R).float().to(device=kpts0.device)
        t_torch = torch.from_numpy(tvec.flatten()).float().to(device=kpts0.device)
        T_list.append(Pose.from_Rt(R_torch, t_torch))

    return T_list



def epipolar_loss(kpts0, kpts1, matches0, T0to1, cam0, cam1, weight=1.0):
    """
    Computes mean squared epipolar residual for valid matches.

    kpts0, kpts1: [B=32, N, 2]
    matches0: [B, N] → indices in kpts1 or -1
    pose0, pose1: [B, 4, 4] ground truth poses
    K: [B, 3, 3]
    0 is follower, 1 is leader
    """
   
    K0 = build_K_matrix(cam0.f, cam0.c)
    K1 = build_K_matrix(cam1.f, cam1.c)

    B = kpts0.shape[0]
    losses = torch.zeros(B, device=kpts0.device, dtype=kpts0.dtype)
    for b in range(B):
        m0 = matches0[b]  # [N]
        valid = m0 > -1
        x0 = kpts0[b][valid]  # [M, 2]  # TODO: Epipolar calculation based on only valid matches or every???
        x1 = kpts1[b][m0[valid]]  # [M, 2]

        if x0.shape[0] < 8:
            continue

        # Normalize to camera coordinates
        K0_inv = torch.inverse(K0[b])
        K1_inv = torch.inverse(K1[b])
        x0_h = F.pad(x0, (0, 1), value=1.0)
        x1_h = F.pad(x1, (0, 1), value=1.0)

        x0_cam = (K0_inv @ x0_h.T).T  # [M, 3]
        x1_cam = (K1_inv @ x1_h.T).T  # [M, 3]

        # Relative pose: T = T1 * T0^-1
        T = T0to1[b]
        R = T0to1[b].R
        t = T0to1[b].t

        # Essential matrix
        tx = torch.tensor([
            [0, -t[2], t[1]],
            [t[2], 0, -t[0]],
            [-t[1], t[0], 0]
        ], device=t.device)

        E = tx @ R

        # Epipolar residuals
        r = torch.einsum('bi,ij,bj->b', x1_cam, E, x0_cam)  # [M]
        losses[b] = weight * (r ** 2).mean()

    return losses

def sampson_epipolar_loss(kpts0, kpts1, matches0, T0to1, cam0, cam1, weight=1.0):
    """
    Computes Sampson epipolar error per batch element.
    
    Returns:
        losses: [B] Sampson errors per batch sample
    """
    K0 = build_K_matrix(cam0.f, cam0.c)
    K1 = build_K_matrix(cam1.f, cam1.c)

    B = kpts0.shape[0]
    losses = torch.zeros(B, device=kpts0.device, dtype=kpts0.dtype)

    for b in range(B):
        m0 = matches0[b]  # [N]
        valid = m0 > -1
        if valid.sum() < 8:
            losses[b] = 0.0
            continue

        x0 = kpts0[b][valid]  # [M, 2]
        x1 = kpts1[b][m0[valid]]  # [M, 2]

        x0_h = F.pad(x0, (0, 1), value=1.0)  # [M, 3]
        x1_h = F.pad(x1, (0, 1), value=1.0)  # [M, 3]

        # Normalize to camera coordinates
        K0_inv = torch.inverse(K0[b])
        K1_inv = torch.inverse(K1[b])
        x0_cam = (K0_inv @ x0_h.T).T  # [M, 3]
        x1_cam = (K1_inv @ x1_h.T).T  # [M, 3]

        # Get relative pose
        if isinstance(T0to1, list) or hasattr(T0to1[b], 'R'):
            R = T0to1[b].R  # [3, 3]
            t = T0to1[b].t  # [3]
        else:
            T = T0to1[b]
            R = T[:3, :3]
            t = T[:3, 3]

        # Essential matrix E = [t]_x R
        tx = torch.tensor([
            [0, -t[2], t[1]],
            [t[2], 0, -t[0]],
            [-t[1], t[0], 0]
        ], device=t.device, dtype=t.dtype)
        E = tx @ R  # [3, 3]

        # Compute numerator: (x1^T E x0)^2
        Ex0 = (E @ x0_cam.T).T  # [M, 3]
        Etx1 = (E.T @ x1_cam.T).T  # [M, 3]
        x1_E_x0 = torch.einsum('ni,ni->n', x1_cam, Ex0)  # [M]
        numerator = x1_E_x0 ** 2  # [M]

        # Compute denominator: sum of squared derivatives
        denom = Ex0[:, 0] ** 2 + Ex0[:, 1] ** 2 + Etx1[:, 0] ** 2 + Etx1[:, 1] ** 2  # [M]
        sampson_error = numerator / (denom + 1e-8)  # [M]

        losses[b] = weight * sampson_error.mean()

    return losses  # Shape: [B]