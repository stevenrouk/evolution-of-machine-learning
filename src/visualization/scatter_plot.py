from bokeh.io import output_notebook, show, export_svgs, export_png
from bokeh.models import ColumnDataSource, HoverTool, ContinuousColorMapper
from bokeh.plotting import figure
#from bokeh.sampledata.autompg import autompg_clean as df
from bokeh.palettes import Spectral6, Dark2
from bokeh.transform import factor_cmap
import matplotlib.cm as cm
import numpy as np
from sklearn.manifold import TSNE


def get_colors(x):
    # get a colormap from matplotlib
    colormap = cm.get_cmap("coolwarm") #choose any matplotlib colormap here

    # define maximum and minimum for cmap
    # colorspan = [-50, 60]

    # create a color channel with a value between 0 and 1
    # outside the colorspan the value becomes 0 (left) and 1 (right)
    # cmap_input = np.interp(x, colorspan, [0, 1], left=0, right=1)

    x = x - min(x)
    x = x / max(x)

    # use colormap to generate rgb-values
    # second value is alfa (not used)
    # third parameter gives int if True, otherwise float
    A_color = colormap(x, 1, True)

    # convert to hex to fit to bokeh
    bokeh_colors = ["#%02x%02x%02x" % (r, g, b) for r, g, b in A_color[:,0:3]]

    return bokeh_colors

    # # create the plot-
    # p = bk.figure(title="Example of importing colormap from matplotlib")

    # p.scatter(x, y, radius=radii,
    #         fill_color=bokeh_colors, fill_alpha=0.6,
    #         line_color=None)


def get_two_topic_scatterplot(df, x_col, y_col, color_col=None):
    # df.cyl = df.cyl.astype(str)
    # df.yr = df.yr.astype(str)

    # group = df.groupby(by=['cyl', 'mfr'])
    # source = ColumnDataSource(group)

    p = figure()

    # p = figure(plot_width=800, plot_height=800, title="Mean MPG by # Cylinders and Manufacturer",
    #         x_range=group, toolbar_location=None, tools="")
    
    #p.scatter(df.mpg, df.hp)
    x = df[x_col].values
    y = df[y_col].values
    if not color_col:
        colors = get_colors(x)
    else:
        colors = get_colors(df[color_col])
    df['color'] = colors
    p.scatter(x=x_col, y=y_col, radius=1.5, fill_alpha=0.5, fill_color='color', source=df)

    # p.xgrid.grid_line_color = None
    # p.xaxis.axis_label = "Manufacturer grouped by # Cylinders"
    p.xaxis.major_label_orientation = 1.2

    # index_cmap = factor_cmap('cyl_mfr', palette=['#2b83ba', '#abdda4', '#ffffbf', '#fdae61', '#d7191c'], 
    #                         factors=sorted(df.cyl.unique()), end=1)

    # p.vbar(x='cyl_mfr', top='mpg_mean', width=1, source=source,
    #     line_color="white", fill_color=index_cmap, 
    #     hover_line_color="darkgrey", hover_fill_color=index_cmap)

    # p.add_tools(HoverTool(tooltips=[("MPG", "@mpg_mean"), ("Cyl, Mfr", "@cyl_mfr")]))

    return p

def get_tsne_scatterplot(df, x_col, y_col, title=None, color_col=None):
    # df.cyl = df.cyl.astype(str)
    # df.yr = df.yr.astype(str)

    # group = df.groupby(by=['cyl', 'mfr'])
    # source = ColumnDataSource(group)

    p = figure(title=title)

    # p = figure(plot_width=800, plot_height=800, title="Mean MPG by # Cylinders and Manufacturer",
    #         x_range=group, toolbar_location=None, tools="")
    
    #p.scatter(df.mpg, df.hp)
    x = df[x_col].values
    y = df[y_col].values
    if not color_col:
        colors = get_colors(x)
    else:
        colors = get_colors(df[color_col])
    df['color'] = colors
    p.scatter(x=x_col, y=y_col, radius=1, fill_alpha='topic_loadings', line_alpha=0, fill_color='color', source=df)

    # p.xgrid.grid_line_color = None
    # p.xaxis.axis_label = "Manufacturer grouped by # Cylinders"
    p.xaxis.major_label_orientation = 1.2

    # index_cmap = factor_cmap('cyl_mfr', palette=['#2b83ba', '#abdda4', '#ffffbf', '#fdae61', '#d7191c'], 
    #                         factors=sorted(df.cyl.unique()), end=1)

    # p.vbar(x='cyl_mfr', top='mpg_mean', width=1, source=source,
    #     line_color="white", fill_color=index_cmap, 
    #     hover_line_color="darkgrey", hover_fill_color=index_cmap)

    # p.add_tools(HoverTool(tooltips=[("MPG", "@mpg_mean"), ("Cyl, Mfr", "@cyl_mfr")]))
    p.add_tools(HoverTool(tooltips=[("Titles", "@titles")]))

    return p

