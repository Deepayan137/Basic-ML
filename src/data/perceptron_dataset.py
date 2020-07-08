class Dataset(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.nSamples = len(X)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def batchify(data=None, batch_size=None, shuffle=False):
    l = len(data)
    n = batch_size
    for ndx in range(0, l, n):
        yield data[ndx:min(ndx + n, l)]
