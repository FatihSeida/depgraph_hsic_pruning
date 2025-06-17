import matplotlib.pyplot as plt

def heatmap(data, *a, **k):
    cmap = k.get('cmap')
    plt.imshow(data, cmap=cmap)
