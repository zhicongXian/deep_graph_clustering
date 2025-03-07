# Created by zhicong.xian at 14:49 07.03.2025 using PyCharm
import scipy as sp

x_filepath = "../datasets/affinity_matrix_from_senet_sparse_20000.npz"
x = sp.sparse.load_npz(x_filepath).toarray()

import plotly.express as px
import plotly

fig = px.imshow(x[:1000])
plotly.offline.plot(fig, filename="affinity_matrix_from_senet_sparse_20000_ssc_heatmap" + '.html', auto_open=True)
