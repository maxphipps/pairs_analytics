from bokeh.plotting import figure
from bokeh.layouts import column, gridplot
from bokeh.models import CustomJS, ColumnDataSource, Slider, Band
from bokeh.models.layouts import Column
from bokeh.palettes import Spectral6
import numpy as np

from utils.math_utils import rolling_mean

"""
Main plot routines
"""

UPDATE_PRICE_CURVE_JS = """
var data = source_pair_fac.data;
var f = slider_pair_fac.value;
var y0 = data['y0'];
var y1 = data['y1'];
var y1_times_f = data['y1_times_f'];
var y_residue = data['y_residue'];
var y_residue_ma = data['y_residue_ma'];

var i_ma = 0;
var ndays_ma = 180;

for (var i = 0; i < y1.length; i++) {
    // update the stock prices scaled by factor
    y1_times_f[i] = y1[i]*f;
    
    // update the residue values
    y_residue[i] = y1_times_f[i] - y0[i];
    
    // update the residue values' moving average
    if (i >= ndays_ma-1) {
        y_residue_ma[i] = 0;
        // cumsum the previous n days
        for (var j = Math.max(0,i-ndays_ma); j <= i; j++) {
            y_residue_ma[i] += y_residue[j];
        }
        y_residue_ma[i] /= ndays_ma;
    }
}

// necessary because we mutated source_pair_fac.data inplace
source_pair_fac.change.emit();
"""


def datetime(x):
    return np.array(x, dtype=np.datetime64)


def gen_dashboard(ticker_tuples: tuple[str, dict]) -> Column:
    """
    Generates the bokeh Column layout
    :param ticker_tuples: Tuples of price data, indexed by ticker
    :return:
    """
    plot_options = dict(width=500, plot_height=300, tools='pan,wheel_zoom')
    colors = Spectral6

    # unzip ticker_tuples
    all_ticker_name, all_ticker_data = zip(*ticker_tuples)
    # TODO: Whilst relying on test data, assert date indexes for the pair are identical.
    #  Update to use yfinance or other data source, and perform inner join of the pairs price frames on their date indexes.
    if all_ticker_data[0]['date'] != all_ticker_data[1]['date']:
        raise RuntimeError('Ticker indexes do not match')

    # initial slider value setting
    s_value = 3.66

    _init_y0 = all_ticker_data[0]['adj_close']
    _init_y1 = all_ticker_data[1]['adj_close']

    # initial values, for dynamic components
    _init_y1_times_f = np.multiply(_init_y1, s_value)
    _init_y_residue = _init_y1_times_f - _init_y0

    # window size for price delta moving average
    window_size = 30*6

    # x axis values
    x_data = datetime(all_ticker_data[0]['date'])

    # container for prices data
    data_dict_prices = dict(x_data=x_data,
                            x_zeros = [0]*len(x_data),
                            y0=_init_y0,
                            y1=_init_y1,
                            y1_times_f=_init_y1_times_f,
                            y_residue=_init_y_residue,
                            y_residue_ma=rolling_mean(_init_y_residue, window_size))

    '''
    Slider for pairs multiplier factor
    '''

    # data container
    source_pair_fac = ColumnDataSource(data=data_dict_prices)

    # TODO: intelligent calculation of initial slider parameters
    s_start = 0.1
    s_end = 10.
    s_step = (s_end - s_start) / 50
    slider_pair_fac = Slider(start=s_start, end=s_end, value=s_value, step=s_step, title="Pairs Price Factor")

    # Slider: updates y1_times_f = y1 * factor
    update_price_curve = CustomJS(args=dict(source_pair_fac=source_pair_fac,
                                            slider_pair_fac=slider_pair_fac),
                                  code=UPDATE_PRICE_CURVE_JS)
    slider_pair_fac.js_on_change('value', update_price_curve)

    '''
    Plot prices (is target of slider)
    '''

    plot_prices = figure(x_axis_type="datetime", title="Stock Closing Prices",
                         **plot_options)

    # 1st ticker
    plot_prices.line("x_data", "y0",
                     source=source_pair_fac,
                     muted_alpha=0.2,
                     color=colors[0],
                     legend_label=all_ticker_name[0])
    # 2nd ticker
    plot_prices.line("x_data", "y1_times_f",
                     muted_alpha=0.2,
                     source=source_pair_fac,
                     color=colors[1],
                     legend_label=all_ticker_name[1])

    plot_prices.grid.grid_line_alpha = 0.3
    plot_prices.xaxis.axis_label = 'Date'
    plot_prices.yaxis.axis_label = 'Price'
    plot_prices.legend.location = "top_left"
    plot_prices.legend.click_policy="mute"

    '''
    Plot residue (is target of slider)
    '''

    # linked x range with plot_prices
    plot_residue = figure(x_axis_type="datetime", title="Pair Price Delta",
                          x_range=plot_prices.x_range,
                          **plot_options)

    # horizontal line
    from bokeh.models import Span
    hline = Span(location=0, dimension='width', line_color='black', line_width=1)
    plot_residue.renderers.extend([hline])

    # the residue data moving average
    plot_residue.line("x_data", "y_residue_ma",
                      source=source_pair_fac,
                      color='gray',
                      muted_alpha=0.2,
                      alpha=0.5,
                      line_width=1.5,
                      legend_label=f'{window_size}D MA')

    # the residue data
    plot_residue.line("x_data", "y_residue",
                      source=source_pair_fac,
                      color=colors[0],
                      legend_label='Price Delta')

    band = Band(base='x_data', lower='x_zeros', upper='y_residue', source=source_pair_fac, level='underlay',
                fill_alpha=0.2, fill_color='#55FF88')
    plot_residue.add_layout(band)

    plot_residue.grid.grid_line_alpha = 0.3
    plot_residue.xaxis.axis_label = 'Date'
    plot_residue.yaxis.axis_label = 'Price'
    plot_residue.legend.location = 'bottom_left'

    p = gridplot([[slider_pair_fac],
                  [column(plot_prices,plot_residue)]])

    return p
