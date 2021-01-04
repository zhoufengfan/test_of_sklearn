import numpy as np
from sklearn import random_projection

if __name__ == '__main__':
    x = np.random.rand(2, 5000)
    # In my opinion, it's better not to set the `n_components` parameter to enable the `SparseRandomProjection` to
    # use `Johnson-Lindenstrauss lemma` to calculate the output dimension. transformer =
    # random_projection.SparseRandomProjection(n_components=2)
    transformer = random_projection.SparseRandomProjection()
    x2 = transformer.fit_transform(x)
    print("x2.shape is {}".format(x2.shape))
