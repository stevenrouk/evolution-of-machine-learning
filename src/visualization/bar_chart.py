from bokeh.io import output_notebook, show, export_svgs, export_png
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure
from bokeh.sampledata.autompg import autompg_clean as df
from bokeh.transform import factor_cmap
import matplotlib.cm as cm

def get_barchart(df, x_col, y_col, title, bar_color, hover_fill_color, y_axis_label=''):
    p = figure(plot_width=800, plot_height=300, title=title,
               x_range=df[x_col], toolbar_location=None, tools="", y_axis_label=y_axis_label)

    p.xgrid.grid_line_color = None
    # p.xaxis.axis_label = "Manufacturer grouped by # Cylinders"
    p.xaxis.major_label_orientation = 1.2

    # index_cmap = factor_cmap('cyl_mfr', palette=['#2b83ba', '#abdda4', '#ffffbf', '#fdae61', '#d7191c'], 
    #                         factors=sorted(df.cyl.unique()), end=1)

    p.vbar(x=x_col, top=y_col, width=1, source=df,
           line_color='black', fill_color=bar_color,
           hover_line_color='black', hover_fill_color=hover_fill_color)

    # p.vbar(x='cyl_mfr', top='mpg_mean', width=1, source=source,
    #     line_color="white", fill_color=index_cmap, 
    #     hover_line_color="darkgrey", hover_fill_color=index_cmap)

    # p.add_tools(HoverTool(tooltips=[("MPG", "@mpg_mean"), ("Cyl, Mfr", "@cyl_mfr")]))
    p.add_tools(HoverTool(tooltips=[("", "@x"), ("", "@y")], show_arrow=False))

    return p
