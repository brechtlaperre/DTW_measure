import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.offsetbox import AnchoredText

def plot_set_of_storms(truth, pred, lookup, dateset, times, save=True, fname='figures/notebook_storms'):
    sns.set(context='paper', style='whitegrid')
    fig, axes = plt.subplots(len(dateset), len(times), sharey=True, figsize=(16,12))
    markers = {'Truth': 'o', 'Pred': 'v'}
    dash = [True, False]
    for j, dates in enumerate(dateset):
        for i, t in enumerate(times):
            ax = axes[j, i]
            start, finish = get_range(lookup[:,t], dates[0], dates[1])
            data = pd.DataFrame(data=truth[start:finish, t], index=lookup[start:finish, t], columns=['Truth'])
            data['Pred'] = pred[start:finish, t]
            #colors = ['#f1a340', '#998ec3']
            sns.lineplot(data=data, ax=ax, markers=markers, palette=sns.cubehelix_palette(2, start=2, rot=.5, dark=.3, light=.5, reverse=True))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m %H:00'))
            if j == 0:
                ax.set_title('Forecasting horizon t+{}h'.format(t+1), fontsize=18)
            elif j == 2:
                ax.set_xlabel('Time (h)')
            if i == 0:
                ax.set_ylabel('Dst (nT)')
            ax.legend(loc="lower left")
            
            #text_box = AnchoredText(dates[0].split('-')[0], frameon=False, loc=4, pad=0.5)
            #ax.add_artist(text_box)
            #plt.setp(text_box.patch, facecolor='white', alpha=0.5)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    plt.tight_layout()

    if save:
        plt.savefig('{}.png'.format(fname), format='png', dpi=300)
        plt.savefig('{}.eps'.format(fname), format='eps')
    else:
        plt.show()

    #a = input('> Save figure? ')
    #if a == 'no' or a == '':
    #    plt.close()
    #else:
    #    plt.savefig('figures/storm_plots/{}.png'.format(a), format='png')
    #    plt.savefig('figures/storm_plots/{}.eps'.format(a), format='eps')
    #    plt.close()

def get_range(lookup, stdate, findate):
    frame = pd.DataFrame(data=np.arange(len(lookup)), index=lookup)
    res = frame[stdate:findate]
    if res.empty:
        print('Month not found')
        return 0, 0
    else:
        return res[0][0], res[0][-1]