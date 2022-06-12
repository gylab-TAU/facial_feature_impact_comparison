from typing import Dict, List
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .stats import calc_graph_measurements


def plot_rocs(dist_lists: Dict[str, pd.DataFrame], fig_titles: List[str], line_names: List[str], rows: List[int], cols: List[int]):
    line_designs = [
        dict(color='royalblue', width=2),
        dict(color='firebrick', width=2),
        dict(color='royalblue', width=2, dash='dot'),
        dict(color='firebrick', width=2, dash='dot'),
    ]

    rocs = make_subplots(rows=5, cols=4,
                         x_title='Dataset', y_title='Model domain',
                         subplot_titles=fig_titles,
                         vertical_spacing=0.05,
                         row_heights=5*[1], column_widths=4*[1])
    for i, key in enumerate(dist_lists):
        curr_list = dist_lists[key]
        fpr, tpr, thresh, roc_auc = calc_graph_measurements(curr_list, 'same', 'cos')
        name = line_names[i]
        rocs.add_trace(go.Scatter(x=fpr, y=tpr, name=f'{name}. AUC={roc_auc}', mode='lines', line=line_designs[i%2]),
            row=1+rows[i], col=1+cols[i])
    for i in range(5):
        for j in range(4):
            rocs.add_shape(
                type='line', line=dict(dash='dash'),
                x0=0, x1=1, y0=0, y1=1, row=1+i, col=1+j) #
#     rocs.layout.annotations[i].update(text=f'{titles[i]}. AUC={roc_auc}')

    rocs.update_layout(height=1250, width=1500, template='plotly_white')
    rocs.update_layout(template='plotly_white')
    rocs.show()
    # rocs.write_html('/home/ssd_storage/experiments/birds/inversion/all_comparisons_ROCs_with_individual_birds_with_vgg19_faces_16022022.html')


def plot_rdms(dfs: List[pd.DataFrame], rows: List[int], cols: List[int], row_titles: List[str], col_titles: List[str]) -> go.Figure:
    rdm_size = 300
    n_rows = 1 + max(rows)
    n_cols = max(cols) + 1
    rdms = make_subplots(rows=n_rows, cols=n_cols,
                         row_titles=row_titles, column_titles=col_titles,
                         vertical_spacing=0.05,
                         column_widths=n_cols*[100], row_heights=n_rows*[100])
    for i, title in enumerate(dfs):
        rdms.add_trace(go.Heatmap(z=dfs[i]),
            row=1+rows[i], col=1+cols[i])
    rdms.update_layout(height=rdm_size * n_rows, width=rdm_size * n_cols,  template='plotly_white')
    return rdms
