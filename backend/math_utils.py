import numpy as np
import math

def get_mid_p(kp):
    num_vertebrae = len(kp) // 4
    mid_p_v = np.zeros((num_vertebrae, 2))
    mid_p = np.zeros((num_vertebrae * 2, 2))
    left_mids = []
    right_mids = []
    
    # First compute the centroid of each vertebra to determine local spine direction
    centers = []
    for i in range(num_vertebrae):
        kp_v = kp[i * 4: (i + 1) * 4]
        cx = sum([p[0] for p in kp_v]) / 4.0
        cy = sum([p[1] for p in kp_v]) / 4.0
        centers.append(np.array([cx, cy]))

    for i in range(num_vertebrae):
        kp_v = np.array(kp[i * 4: (i + 1) * 4])
        mid_p_v[i] = centers[i]
        
        # Determine local spine curve direction D
        if num_vertebrae >= 2:
            if i == 0:
                D = centers[1] - centers[0]
            elif i == num_vertebrae - 1:
                D = centers[i] - centers[i - 1]
            else:
                D = centers[i + 1] - centers[i - 1]
        else:
            D = np.array([0.0, 1.0])
            
        if np.linalg.norm(D) < 1e-5:
            D = np.array([0.0, 1.0])
        D = D / np.linalg.norm(D)
        
        # Sort the 4 corners clockwise around the center
        c = centers[i]
        angles = np.arctan2(kp_v[:, 1] - c[1], kp_v[:, 0] - c[0])
        cw_order = np.argsort(angles)
        V = kp_v[cw_order]
        
        # Option A: Top endplate is (V[0], V[1]), Bottom endplate is (V[2], V[3])
        u1 = (V[2] + V[3])/2.0 - (V[0] + V[1])/2.0
        norm_u1 = np.linalg.norm(u1) if np.linalg.norm(u1) > 1e-5 else 1.0
        score1 = abs(np.dot(u1, D)) / norm_u1
        
        # Option B: Top endplate is (V[1], V[2]), Bottom endplate is (V[3], V[0])
        u2 = (V[3] + V[0])/2.0 - (V[1] + V[2])/2.0
        norm_u2 = np.linalg.norm(u2) if np.linalg.norm(u2) > 1e-5 else 1.0
        score2 = abs(np.dot(u2, D)) / norm_u2
        
        if score1 >= score2:
            if np.dot(u1, D) >= 0:
                p1, p2, p3, p4 = V[0], V[1], V[3], V[2]
            else:
                p1, p2, p3, p4 = V[2], V[3], V[1], V[0]
        else:
            if np.dot(u2, D) >= 0:
                p1, p2, p3, p4 = V[1], V[2], V[0], V[3]
            else:
                p1, p2, p3, p4 = V[3], V[0], V[2], V[1]
                
        # Update kp array in place so semantic labels (Point 1..Point 4) are completely standardized
        kp[i * 4]     = [float(p1[0]), float(p1[1])]
        kp[i * 4 + 1] = [float(p2[0]), float(p2[1])]
        kp[i * 4 + 2] = [float(p3[0]), float(p3[1])]
        kp[i * 4 + 3] = [float(p4[0]), float(p4[1])]
        
        # Calculate midpoints according to user formula:
        # first midpoint = ((x1 + x3)/2, (y1 + y3)/2)
        # second midpoint = ((x2 + x4)/2, (y2 + y4)/2)
        lm = [(p1[0] + p3[0]) / 2.0, (p1[1] + p3[1]) / 2.0]
        rm = [(p2[0] + p4[0]) / 2.0, (p2[1] + p4[1]) / 2.0]
        
        left_mids.append(lm)
        right_mids.append(rm)
        
        mid_p[i * 2]     = [(p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0]
        mid_p[i * 2 + 1] = [(p3[0] + p4[0]) / 2.0, (p3[1] + p4[1]) / 2.0]

    return mid_p_v, mid_p, left_mids, right_mids

def _isS(mid_p_v):
    n = len(mid_p_v)
    if n < 3: return "C"
    p0 = mid_p_v[0]
    pn = mid_p_v[n - 1]
    ll = np.zeros(n)
    for i in range(n):
        if p0[1] - pn[1] != 0 and p0[0] - pn[0] != 0:
            ll[i] = (mid_p_v[i][1] - pn[1]) / (p0[1] - pn[1]) - (mid_p_v[i][0] - pn[0]) / (p0[0] - pn[0])
    ll_inner = ll[1:-1]
    signs = np.sign(ll_inner[np.abs(ll_inner) > 1e-4])
    if len(signs) == 0: return "C"
    all_same_sign = np.all(signs > 0) or np.all(signs < 0)
    return "C" if all_same_sign else "S"

def cobb_angle_cal(landmark_xy, image_shape=None):
    # Robust flattening and conversion to [[x, y], ...]
    kp = []
    if len(landmark_xy) > 0:
        if not isinstance(landmark_xy[0], list):
            # Case 1: Flat list [x1, y1, x2, y2...] or [x1..xn, y1..yn]
            # Check for [x1..xn, y1..yn] format (split by half)
            n = len(landmark_xy) // 2
            kp = [[float(landmark_xy[i]), float(landmark_xy[i + n])] for i in range(n)]
        elif isinstance(landmark_xy[0][0], list):
            # Case 2: Nested by vertebra [[[x,y], [x,y], [x,y], [x,y]], ...]
            for v in landmark_xy:
                for pt in v:
                    kp.append([float(pt[0]), float(pt[1])])
        else:
            # Case 3: Already a list of points [[x,y], [x,y], ...]
            kp = [[float(pt[0]), float(pt[1])] for pt in landmark_xy]
        
    num_vertebrae = len(kp) // 4
    if num_vertebrae < 2:
        return [0, 0, 0], {"pt": {"angle": 0, "idxs": [0,0]}, "mt": {"angle": 0, "idxs": [0,0]}, "tl": {"angle": 0, "idxs": [0,0]}}, "C", [], kp

    # Group keypoints into vertebrae, compute centroid Y, and sort top-to-bottom
    vertebrae = []
    for i in range(num_vertebrae):
        v_pts = kp[i * 4 : (i + 1) * 4]
        cy = sum([p[1] for p in v_pts]) / 4.0
        vertebrae.append((cy, v_pts))
    
    vertebrae.sort(key=lambda item: item[0])
    
    # Flatten back into kp list in perfect sorted order
    kp = []
    for cy, v_pts in vertebrae:
        kp.extend(v_pts)

    mid_p_v, mid_p, left_mids, right_mids = get_mid_p(kp)
    
    vec_m = np.zeros((num_vertebrae, 2))
    for i in range(num_vertebrae):
        lm = left_mids[i]
        rm = right_mids[i]
        vec_m[i] = [rm[0] - lm[0], rm[1] - lm[1]]

    angles_matrix = np.zeros((num_vertebrae, num_vertebrae))
    for i in range(num_vertebrae):
        for j in range(num_vertebrae):
            A, B = vec_m[i], vec_m[j]
            norm_A, norm_B = np.linalg.norm(A), np.linalg.norm(B)
            if norm_A == 0 or norm_B == 0: continue
            cos_theta = np.dot(A, B) / (norm_A * norm_B)
            angles_matrix[i][j] = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    
    curve_type = _isS(mid_p_v)
    
    pt_angle, mt_angle, tl_angle = 0.0, 0.0, 0.0
    pt_idxs, mt_idxs, tl_idxs = [0, 0], [0, 0], [0, 0]
    
    if num_vertebrae >= 3:
        # PT: T1 to T4 (Indices 0 to 3)
        pt_end = min(3, num_vertebrae - 1)
        # MT: T4 to T12 (Indices 3 to 11)
        mt_start = min(3, num_vertebrae - 1)
        mt_end = min(11, num_vertebrae - 1)
        # LT: T12 to T16/T17 (Indices 11 onwards)
        lt_start = min(11, num_vertebrae - 1)
        
        r1_matrix = angles_matrix[0:pt_end+1, 0:pt_end+1]
        if r1_matrix.size > 0:
            pt_angle = np.max(r1_matrix)
            p2, p1 = np.unravel_index(np.argmax(r1_matrix), r1_matrix.shape)
            if p1 > p2: p1, p2 = p2, p1
            pt_idxs = [int(p1), int(p2)]
            
        r2_matrix = angles_matrix[mt_start:mt_end+1, mt_start:mt_end+1]
        if r2_matrix.size > 0:
            mt_angle = np.max(r2_matrix)
            p2, p1 = np.unravel_index(np.argmax(r2_matrix), r2_matrix.shape)
            if p1 > p2: p1, p2 = p2, p1
            mt_idxs = [int(mt_start + p1), int(mt_start + p2)]
            
        r3_matrix = angles_matrix[lt_start:num_vertebrae, lt_start:num_vertebrae]
        if r3_matrix.size > 0:
            lt_angle = np.max(r3_matrix)
            p2, p1 = np.unravel_index(np.argmax(r3_matrix), r3_matrix.shape)
            if p1 > p2: p1, p2 = p2, p1
            tl_idxs = [int(lt_start + p1), int(lt_start + p2)]
    else:
        mt_angle = np.max(angles_matrix)
        p2, p1 = np.unravel_index(np.argmax(angles_matrix), angles_matrix.shape)
        if p1 > p2: p1, p2 = p2, p1
        mt_idxs = [int(p1), int(p2)]

    angles_with_pos = {
        "pt": {"angle": float(pt_angle), "idxs": [int(pt_idxs[0]), int(pt_idxs[1])]},
        "mt": {"angle": float(mt_angle), "idxs": [int(mt_idxs[0]), int(mt_idxs[1])]},
        "lt": {"angle": float(lt_angle), "idxs": [int(tl_idxs[0]), int(tl_idxs[1])]}
    }
    
    midpoint_lines = []
    for i in range(num_vertebrae):
        lm = left_mids[i]
        rm = right_mids[i]
        midpoint_lines.append([[float(lm[0]), float(lm[1])], [float(rm[0]), float(rm[1])]])

    return [float(pt_angle), float(mt_angle), float(lt_angle)], angles_with_pos, curve_type, midpoint_lines, kp
