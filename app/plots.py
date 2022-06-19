from bokeh.plotting import figure
from bokeh.layouts import column, gridplot
from bokeh.models import CustomJS, ColumnDataSource, Slider, Band, Span
from bokeh.models.layouts import Column
from bokeh.models.ranges import DataRange1d
from bokeh.palettes import Spectral6
import numpy as np
import pandas as pd

from app.utils.math_utils import rolling_mean

"""
Main plot routines
"""

UPDATE_PRICE_CURVE_JS = """
var data = data_container.data;
var f = pair_factor_slider.value;
var y0 = data['y0'];
var y1 = data['y1_unscaled'];
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

// necessary because we mutated data_container.data inplace
data_container.change.emit();
"""


def datetime(x):
    return np.array(x, dtype=np.datetime64)


class Dashboard:
    PRICE_DELTA_MA_WINDOW_DAYS = 30 * 6
    INITIAL_SLIDER_VALUE = 2.87
    COLORS = Spectral6

    def __init__(self, df_prices: pd.DataFrame):
        """
        Dashboard generator
        :param df_prices: Dataframe of price data, indexed by ticker label
        """
        self.plot_options = dict(width=500, plot_height=300, tools='pan,wheel_zoom')
        # Construct data container
        data_container = self._generate_data_container(df_prices)
        # Gather plot components
        slider_pair_fac = self._construct_slider(data_container)
        price_plot = self._construct_price_plot(data_container, df_prices.columns)
        # residue plot x range linked with prices plot
        residue_plot = self._construct_residue_plot(data_container, x_axis_link=price_plot.x_range)
        # Construct plot
        self.p_layout = gridplot([[slider_pair_fac],
                                  [column(price_plot, residue_plot)]])

    def _generate_data_container(self, df_prices: pd.DataFrame) -> ColumnDataSource:
        """
        Generates the data required for plotting
        :param df_prices:
        :return:
        """
        y0 = df_prices.iloc[:, 0].values
        y1_unscaled = df_prices.iloc[:, 1].values
        # initial y values, for dynamic components
        y1_times_f = np.multiply(y1_unscaled, self.INITIAL_SLIDER_VALUE)
        y_residue = y1_times_f - y0
        # x axis values
        x_data = datetime(df_prices.index)
        # container for prices data
        data_dict = dict(x_data=x_data,
                         x_zeros=[0] * len(x_data),
                         y0=y0,
                         y1_unscaled=y1_unscaled,
                         y1_times_f=y1_times_f,
                         y_residue=y_residue,
                         y_residue_ma=rolling_mean(y_residue, self.PRICE_DELTA_MA_WINDOW_DAYS))
        return ColumnDataSource(data=data_dict)

    def _construct_slider(self, data_container: ColumnDataSource) -> Slider:
        """
        Constructs slider for pairs multiplier factor
        :return:
        """
        # TODO: intelligent calculation of initial slider parameters
        slider_params = {'start': 0.1,
                         'end': 10.,
                         'value': self.INITIAL_SLIDER_VALUE}
        s_step = (slider_params['start'] - slider_params['end']) / 50
        pair_factor_slider = Slider(**slider_params, step=s_step, title="Pairs Price Factor")

        # Slider updates y1_times_f = y1 * factor
        update_price_curve = CustomJS(args=dict(data_container=data_container,
                                                pair_factor_slider=pair_factor_slider),
                                      code=UPDATE_PRICE_CURVE_JS)
        pair_factor_slider.js_on_change('value', update_price_curve)
        return pair_factor_slider

    def _construct_price_plot(self,
                              data_container: ColumnDataSource,
                              ticker_labels: tuple):
        """
        Constructs price plot (target of slider)
        :return:
        """
        price_plot = figure(x_axis_type="datetime", title="Stock Closing Prices",
                             **self.plot_options)
        # Plot prices
        price_plot.line("x_data", "y0",
                        source=data_container,
                        muted_alpha=0.2,
                        color=self.COLORS[0],
                        legend_label=ticker_labels[0])
        price_plot.line("x_data", "y1_times_f",
                        muted_alpha=0.2,
                        source=data_container,
                        color=self.COLORS[1],
                        legend_label=ticker_labels[1])

        price_plot.grid.grid_line_alpha = 0.3
        price_plot.xaxis.axis_label = 'Date'
        price_plot.yaxis.axis_label = 'Price'
        price_plot.legend.location = "top_left"
        price_plot.legend.click_policy="mute"
        return price_plot

    def _construct_residue_plot(self,
                                data_container: ColumnDataSource,
                                x_axis_link: DataRange1d):
        """
        Constructs residue plot (target of slider)
        :param x_axis_link: x axis object to couple this plot with
        :return:
        """
        plot_residue = figure(x_axis_type="datetime", title="Pair Price Delta",
                              x_range=x_axis_link,
                              **self.plot_options)

        # horizontal line
        hline = Span(location=0, dimension='width', line_color='black', line_width=1)
        plot_residue.renderers.extend([hline])

        # moving average of residue series
        plot_residue.line("x_data", "y_residue_ma",
                          source=data_container,
                          color='gray',
                          muted_alpha=0.2,
                          alpha=0.5,
                          line_width=1.5,
                          legend_label=f'{self.PRICE_DELTA_MA_WINDOW_DAYS}D MA')

        # residue series
        plot_residue.line("x_data", "y_residue",
                          source=data_container,
                          color=self.COLORS[0],
                          legend_label='Price delta')
        band = Band(base='x_data', lower='x_zeros', upper='y_residue', source=data_container, level='underlay',
                    fill_alpha=0.2, fill_color='#55FF88')
        plot_residue.add_layout(band)

        plot_residue.grid.grid_line_alpha = 0.3
        plot_residue.xaxis.axis_label = 'Date'
        plot_residue.yaxis.axis_label = r'$$\Delta \mathrm{Price}$$'
        plot_residue.legend.location = 'bottom_left'
        return plot_residue

    def get_plot(self) -> Column:
        """
        Returns the bokeh Column layout
        :return:
        """
        return self.p_layout
