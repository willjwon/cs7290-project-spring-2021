import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.compression.compression_tracker import CompressionTracker


def plot_compression_ratio_histogram(tracker: CompressionTracker) -> None:
    # cleanup date
    hist_table = tracker.compression_ratio_histogram
    hist_table_total = sum(hist_table)
    data = pd.DataFrame()

    for index, freq in enumerate(hist_table):
        data.loc[index, 'ratio'] = index / 1000
        data.loc[index, 'frequency'] = freq / hist_table_total

    # dump data
    data.to_csv(f'./result/{tracker.__class__.__name__}.csv')

    # plot figure
    sns.set_style(style='ticks')

    fig, ax = plt.subplots(1, 1)
    sns.lineplot(data=data,
                 x='ratio', y='frequency',
                 ax=ax)

    fig.tight_layout()
    fig.show()
    fig.savefig(f'./result/{tracker.__class__.__name__}.pdf')
    fig.clf()
    plt.close(fig=fig)
