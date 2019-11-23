import numpy as np 


RED_Y = 0.3
GREEN_Y = 0.59
BLUE_Y = 0.11


def rgb2hcy(img):
    flat = img.reshape((-1, 3))
    R, G, B = flat.T

    Y = RED_Y * R + GREEN_Y * G + BLUE_Y * B
    
    M, N = np.max(flat, axis=1), np.min(flat, axis=1)
    C = M - N

    H = np.zeros_like(C)

    # Avoid dividing by zero
    mask = C > 0

    MR = (G[mask] - B[mask])/C[mask] % 6
    MG = (B[mask] - R[mask])/C[mask] + 2
    MB = (R[mask] - G[mask])/C[mask] + 4

    maskR = M[mask] == R[mask]
    maskG = (M[mask] == G[mask]) & (~maskR)
    maskB = (M[mask] == B[mask]) & (~maskG) & (~maskR)

    H[mask] = maskR * MR + maskG * MG + maskB * MB
    H = 60 * H
    
    nimg = np.hstack((H[:, None], C[:, None], Y[:, None]))
    nimg = nimg.reshape(img.shape)
    return nimg


def hcy2rgb(img, clip=True):
    flat = img.reshape((-1, 3))

    H, C, Y = flat.T

    H_p = H / 60

    X = C * (1 - np.abs(H_p % 2 - 1))

    choice = np.vstack((C, X, np.zeros_like(C)))
    _C, _X, _0 = 0, 1, 2
    index = np.array([[_C, _X, _0],
                      [_X, _C, _0],
                      [_0, _C, _X],
                      [_0, _X, _C],
                      [_X, _0, _C],
                      [_C, _0, _X]])
    
    masks = np.array([(i <= H_p) & (H_p < i+1) for i in range(6)])

    R1, G1, B1 = np.einsum("ikj,ij->kj", choice[index], masks)

    m = Y - (RED_Y * R1 + GREEN_Y * G1 + BLUE_Y * B1)
    R, G, B = R1 + m, G1 + m, B1 + m

    nimg = np.hstack((R[:, None], G[:, None], B[:, None]))
    nimg = nimg.reshape(img.shape)

    # avoid bad rounding in C==0
    if clip:
        return np.clip(nimg, 0, 1)
    else:
        return nimg


if __name__ == "__main__":
    import imageio
    img = imageio.imread("lena.png") / 255.
    nimg = rgb2hcy(img)
    oimg = hcy2rgb(nimg)
    assert np.max(np.abs(oimg - img)) < 10**(-13)
