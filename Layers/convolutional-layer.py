import numpy as np

class Conv2d:
    def __init__(self, c_in, c_out, k, pad=0, stride=1):
        self.c_in, self.c_out = c_in, c_out
        self.k, self.p, self.s = k, pad, stride

        # -----Learnable Parameters-----
        self.W = np.random.randn(c_out, c_in, k, k) * np.sqrt(2.0 / (c_in * k * k))
        self.b = np.zeros(c_out)

        # -----Gradients-----
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        self.Xp = None

    def forward(self, X):
        k, s, p = self.k, self.s, self.p
        N, _, H, W = X.shape

        Xp = np.pad(X, ((0, 0), (0, 0), (p, p), (p, p)))

        H_out = (H + 2*p - k) // s + 1
        W_out = (W + 2*p - k) // s + 1

        Y = np.zeros((N, self.c_out, H_out, W_out))

        for n in range(N):
            for f in range(self.c_out):
                for i in range(H_out):
                    for j in range(W_out):
                        patch = Xp[n, :, i*s:i*s+k, j*s:j*s+k]
                        Y[n, f, i, j] = np.sum(patch * self.W[f]) + self.b[f]

        self.Xp = Xp
        return Y

    def backward(self, dY):
        k, s, p = self.k, self.s, self.p
        N, _, H_out, W_out = dY.shape
        Xp = self.Xp

        self.dW[:] = 0
        self.db[:] = 0

        dXp = np.zeros_like(Xp)

        for n in range(N):
            for f in range(self.c_out):
                for i in range(H_out):
                    for j in range(W_out):
                        patch = Xp[n, :, i*s:i*s+k, j*s:j*s+k]
                        g = dY[n, f, i, j]

                        self.dW[f] += g * patch
                        dXp[n, :, i*s:i*s+k, j*s:j*s+k] += self.W[f] * g
                        self.db[f] += g

        H_p, W_p = Xp.shape[2], Xp.shape[3]

        if p > 0:
            return dXp[:, :, p:H_p-p, p:W_p-p]
        else:
            return dXp

def gradient_checking(epsilon=1e-5):
    np.random.seed(42)

    layer = Conv2d(c_in=2, c_out=3, k=3, pad=1, stride=1)
    X = np.random.randn(2, 2, 5, 5)

    Y = layer.forward(X)
    R = np.random.randn(*Y.shape)
    dX_analytical = layer.backward(R)

    dW = np.zeros_like(layer.W)
    db = np.zeros_like(layer.b)
    dX = np.zeros_like(X)

    it_W = np.nditer(layer.W, flags=['multi_index'])
    it_X = np.nditer(X, flags=['multi_index'])

    for _ in it_W:
        idx = it_W.multi_index
        W = layer.W[idx]

        layer.W[idx] = W + epsilon
        WF = np.sum(layer.forward(X) * R)
        layer.W[idx] = W - epsilon
        WI = np.sum(layer.forward(X) * R)

        layer.W[idx] = W

        dW[idx] = (WF - WI) / (2 * epsilon)

    for _ in it_X:
        idx = it_X.multi_index

        x = X[idx]
                
        X[idx] = x + epsilon
        XpF = np.sum(layer.forward(X) * R)
        X[idx] = x - epsilon
        XpI = np.sum(layer.forward(X) * R)

        X[idx] = x

        dX[idx] = (XpF - XpI) / (2 * epsilon)

    for idx in range(layer.b.size):
        b = layer.b[idx]
        
        layer.b[idx] = b + epsilon
        bF = np.sum(layer.forward(X) * R)
        layer.b[idx] = b - epsilon
        bI = np.sum(layer.forward(X) * R)

        layer.b[idx] = b

        db[idx] = (bF - bI) / (2 * epsilon)

    W_err = np.max(np.abs(dW - layer.dW))
    b_err = np.max(np.abs(db - layer.db))
    X_err = np.max(np.abs(dX - dX_analytical))

    print(f"W Error: {W_err}\nb Error: {b_err}\nX Error: {X_err}")

if __name__ == '__main__':
    gradient_checking()