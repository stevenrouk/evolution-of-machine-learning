from bokeh.io import output_notebook, show, export_svgs, export_png
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure
from bokeh.sampledata.autompg import autompg_clean as df
from bokeh.transform import factor_cmap

from sklearn.manifold import TSNE

def get_two_topic_scatterplot(x, y):
    df.cyl = df.cyl.astype(str)
    df.yr = df.yr.astype(str)

    group = df.groupby(by=['cyl', 'mfr'])
    source = ColumnDataSource(group)

    p = figure()

    # p = figure(plot_width=800, plot_height=800, title="Mean MPG by # Cylinders and Manufacturer",
    #         x_range=group, toolbar_location=None, tools="")
    
    #p.scatter(df.mpg, df.hp)
    p.scatter(x, y)

    p.xgrid.grid_line_color = None
    p.xaxis.axis_label = "Manufacturer grouped by # Cylinders"
    p.xaxis.major_label_orientation = 1.2

    # index_cmap = factor_cmap('cyl_mfr', palette=['#2b83ba', '#abdda4', '#ffffbf', '#fdae61', '#d7191c'], 
    #                         factors=sorted(df.cyl.unique()), end=1)

    # p.vbar(x='cyl_mfr', top='mpg_mean', width=1, source=source,
    #     line_color="white", fill_color=index_cmap, 
    #     hover_line_color="darkgrey", hover_fill_color=index_cmap)

    # p.add_tools(HoverTool(tooltips=[("MPG", "@mpg_mean"), ("Cyl, Mfr", "@cyl_mfr")]))

    return p
