import numpy as np

class Maxpool2D:
    def __init__(self, c_in, k, pad=0, stride=1):
        self.c_in = c_in
        self.k = k
        self.pad = pad
        self.stride = stride
        self.Xp = None
        self.idx = np.array([])

    def forward(self, X):
        k, s, p = self.k, self.stride, self.pad
        N, _, H, W = X.shape

        Xp = np.pad(X, ((0, 0), (0, 0), (p, p), (p, p)))

        H_out = (H + 2 * p - k) // s + 1
        W_out = (W + 2 * p - k) // s + 1

        idx = np.zeros((N, self.c_in, H_out, W_out), dtype=int)

        Y = np.zeros((N, self.c_in, H_out, W_out))

        for n in range(N):
            for f in range(self.c_in):
                for i in range(H_out):
                    for j in range(W_out):
                        patch = Xp[n, f, i*s:i*s+k, j*s:j*s+k]
                        Y[n, f, i, j] = np.max(patch)
                        idx[n, f, i, j] = np.argmax(patch)
                        self.idx = idx

        self.Xp = Xp
        return Y

    def backward(self, dY):
        k, s, p = self.k, self.stride, self.pad
        N, _, H_out, W_out = dY.shape
        Xp = self.Xp

        dXp = np.zeros_like(Xp)

        for n in range(N):
            for f in range(self.c_in):
                for i in range(H_out):
                    for j in range(W_out):
                        row, column = np.unravel_index(int(self.idx[n, f, i, j]), (k, k))
                        r = i * s + row
                        c = j * s + column
                        dXp[n, f, r, c] += dY[n, f, i, j]

        return dXp
    
np.random.seed(42)
m1 = Maxpool2D(2, 2)
X = np.random.randn(2, 2, 3, 3)
print(f"X: {X}")
Y = m1.forward(X)
print(f"Y: {Y}")

dY = np.random.randn(2, 2, 2, 2)
dXp = m1.backward(dY)
print(f"dXp: {dXp}")