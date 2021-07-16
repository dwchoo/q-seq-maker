from encoder_decoder.seq_encoder_decoder import one_hot_encoder

import numpy as np
import plotly.graph_objects as go

def graph_seq_distribution(seq_list, title= None, save=False):
    seq_list = np.reshape(seq_list, (-1,))
    assert len(seq_list.shape) == 1, f'Check seq_list shape, {seq_list.shape}'

    seq_one_hot_list = one_hot_encoder.batch_seq_encoder(seq_list)
    distribution_list = np.sum(seq_one_hot_list, axis=0)
    graph_fig = graph_distribution(distribution_list, title, save)


def graph_distribution(distribution_list, title = 'None', save=False):
    seq_distribution_calc = np.array(distribution_list)
    seq_length = len(seq_distribution_calc)
    x_pos_num = np.arange(1,seq_length+1)
    fig = go.Figure(data = [
            go.Bar(name='A',x=x_pos_num,y=seq_distribution_calc[:,0]),
            go.Bar(name='C',x=x_pos_num,y=seq_distribution_calc[:,1]),
            go.Bar(name='G',x=x_pos_num,y=seq_distribution_calc[:,2]),
            go.Bar(name='T',x=x_pos_num,y=seq_distribution_calc[:,3]),
    ])
    fig.update_layout(title=title, title_x = 0.5)
    fig.update_layout(xaxis=dict(title='Position',tickmode='linear'))
    fig.update_layout(bargap=0.5)
    fig.update_layout(barmode='group')
    #if save:
        #fig.write_image(f"{title}.pdf")
    return fig

