# Nov 2024

import matplotlib.pyplot as plt

def plot_setup(title, xlabel, ylabel):
    plt.rcParams.update({'font.size': 13})
    fig, ax = plt.subplots(dpi=600)  # different at home

    ax.minorticks_on()
    ax.tick_params(direction='in', which='both')
    ax.grid(visible=True, axis='both', which='major', linestyle='--', 
            alpha=0.5)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    return fig, ax


def plot_line(ax, x, y, label, colour='none', alpha=1):
    if colour == 'none':
        ax.plot(x, y, label=label, alpha=alpha)
        return
    
    ax.plot(x, y, label=label, color=colour, alpha=alpha)
    

def plot_constellation(ax, points, label='', colour='none', alpha=1, 
                       marker='o', edgecolour='black'):
    if edgecolour=='none':
        ax.scatter(points.real, points.imag, label=label, color=colour,
                   alpha=alpha, marker=marker)
        return
        
    ax.scatter(points.real, points.imag, label=label, color=colour,
               alpha=alpha, marker=marker, edgecolor=edgecolour)
    
    
def plot_show(ax):
    ax.legend()

    plt.show()
    