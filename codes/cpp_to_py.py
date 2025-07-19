import numpy as np
import random
import math

def generate_well_mask(well_positions, grid_shape, well_range=15, sigma=5.0):
    """
    well_positions: list of (line, cmp) tuple
    """
    vWellMask = dict()
    for wp in well_positions:
        for iline in range(-well_range, well_range+1):
            for icmp in range(-well_range, well_range+1):
                line = wp[0] + iline
                cmp_ = wp[1] + icmp
                if not (0 <= line < grid_shape[0] and 0 <= cmp_ < grid_shape[1]):
                    continue
                crd = (line, cmp_)
                weight = math.exp(-(iline**2 + icmp**2) / (2 * sigma**2))
                if crd not in vWellMask or vWellMask[crd] < weight:
                    vWellMask[crd] = weight
    return vWellMask

def calc_grid_pos(grid_shape, points):
    res = []
    def in_grid(line, cmp_):
        return 0 <= line < grid_shape[0] and 0 <= cmp_ < grid_shape[1]
    for i in range(1, len(points)):
        p0 = points[i-1]
        p1 = points[i]
        x0, y0 = int(round(p0[0])), int(round(p0[1]))
        x1, y1 = int(round(p1[0])), int(round(p1[1]))
        dx = abs(x1-x0)
        dy = abs(y1-y0)
        n = max(dx, dy)
        for step in range(n+1):
            t = step / n if n else 0
            x = int(round(x0 + t*(x1-x0)))
            y = int(round(y0 + t*(y1-y0)))
            if in_grid(y, x):
                if len(res) == 0 or res[-1] != (y, x):
                    res.append((y, x))
    return np.array(res, dtype=int)   # shape=(N,2)

def get_wellline_and_mask(well_positions, grid_shape, vWellMask, max_try=100):
    nwell = len(well_positions)
    assert nwell >= 3
    for attempt in range(max_try):
        idx = random.sample(range(nwell), 3)
        wp = [well_positions[i] for i in idx]
        d01 = (wp[0][1]-wp[1][1])**2 + (wp[0][0]-wp[1][0])**2
        d12 = (wp[1][1]-wp[2][1])**2 + (wp[1][0]-wp[2][0])**2
        d20 = (wp[2][1]-wp[0][1])**2 + (wp[2][0]-wp[0][0])**2
        dists = [d01, d12, d20]
        seqs = [
            [1,0,2], [2,1,0], [1,2,0], [2,0,1], [0,2,1]
        ]
        conds = [
            d01 <= d20 and d20 <= d12,
            d12 <= d01 and d01 <= d20,
            d12 <= d20 and d20 <= d01,
            d20 <= d01 and d01 <= d12,
            d20 <= d12 and d12 <= d01
        ]
        for cond, seq in zip(conds, seqs):
            if cond:
                idx = [idx[i] for i in seq]
                wp = [well_positions[i] for i in idx]
                break
        def atan2d(y, x): return math.atan2(y, x)
        M_PI = math.pi
        M_PI_2 = math.pi/2
        n10 = atan2d(wp[0][0]-wp[1][0], wp[0][1]-wp[1][1]) + M_PI_2
        n12 = atan2d(wp[2][0]-wp[1][0], wp[2][1]-wp[1][1]) + M_PI_2
        if n10 > M_PI: n10 -= M_PI*2
        if n12 > M_PI: n12 -= M_PI*2
        if n10-n12 > M_PI: n10 -= M_PI*2
        elif n10-n12 < -M_PI: n12 -= M_PI*2
        L1 = (n12-n10)*random.random() + n10
        if L1 < -M_PI: L1 += M_PI*2
        elif L1 > M_PI: L1 -= M_PI*2
        L01 = atan2d(wp[1][0]-wp[0][0], wp[1][1]-wp[0][1])
        n1 = L1 + M_PI_2
        while L01-n1 > M_PI_2: L01 -= M_PI
        while L01-n1 < -M_PI_2: n1 -= M_PI
        L0 = (n1-L01)*random.random() + L01
        if L0 < -M_PI: L0 += M_PI*2
        elif L0 > M_PI: L0 -= M_PI*2
        L21 = atan2d(wp[1][0]-wp[2][0], wp[1][1]-wp[2][1])
        n1 = L1 + M_PI_2
        while L21-n1 > M_PI_2: L21 -= M_PI
        while L21-n1 < -M_PI_2: n1 -= M_PI
        L2 = (n1-L21)*random.random() + L21
        if L2 < -M_PI: L2 += M_PI*2
        elif L2 > M_PI: L2 -= M_PI*2
        def calc_point(cmp0, line0, cmp1, line1, L0, L1):
            if abs(L1) == M_PI_2:
                return (cmp1, math.tan(L0)*(cmp1-cmp0)+line0)
            elif abs(L0) == M_PI_2:
                return (cmp0, math.tan(L1)*(cmp0-cmp1)+line1)
            else:
                x = (math.tan(L0)*cmp0 - math.tan(L1)*cmp1 + line1 - line0)/(math.tan(L0)-math.tan(L1))
                y = math.tan(L1)*(x-cmp1)+line1
                return (x, y)
        p01 = calc_point(wp[0][1], wp[0][0], wp[1][1], wp[1][0], L0, L1)
        p12 = calc_point(wp[2][1], wp[2][0], wp[1][1], wp[1][0], L2, L1)
        def in_grid_pt(p):
            x, y = p
            return 0 <= int(round(y)) < grid_shape[0] and 0 <= int(round(x)) < grid_shape[1]
        if not (in_grid_pt(p01) and in_grid_pt(p12)):
            continue
        cmp0, cmp2 = wp[0][1], wp[2][1]
        line0, line2 = wp[0][0], wp[2][0]
        x = grid_shape[1]-1 if cmp0 > p01[0] else 0
        y = grid_shape[0]-1 if line0 > p01[1] else 0
        pt0 = (cmp0, y) if abs(L0)==M_PI_2 else (x, math.tan(L0)*(x-cmp0)+line0)
        x = grid_shape[1]-1 if cmp2 > p12[0] else 0
        y = grid_shape[0]-1 if line2 > p12[1] else 0
        pt3 = (cmp2, y) if abs(L2)==M_PI_2 else (x, math.tan(L2)*(x-cmp2)+line2)
        vPoint = [pt0, p01, p12, pt3]
        # 剖面点：返回 shape=(N,2) 的 numpy 数组
        vCrd = calc_grid_pos(grid_shape, vPoint)
        # 掩码数组
        vMask = np.array([vWellMask.get((line, cmp), 0.0) for line, cmp in vCrd])
        return vCrd, vMask
    raise RuntimeError("Failed to generate valid well line after many attempts")

# # 示例
if __name__ == "__main__":
    grid_shape = (100, 100)
    wells = [(20,10), (80,90), (50,50), (30,80)]
    vWellMask = generate_well_mask(wells, grid_shape, well_range=15, sigma=np.sqrt(25))
    vCrd, vMask = get_wellline_and_mask(wells, grid_shape, vWellMask)
    print("vCrd shape:", vCrd.shape)  # (N,2)
    print("vMask shape:", vMask.shape)
    print(vCrd)
    # 用法举例
    mback = np.random.randn(10, 100, 100)
    # 取剖面采样
    implow_train = [mback[:, vCrd[:,0], vCrd[:,1]]]  # shape: (10, N)