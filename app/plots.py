from bokeh.plotting import figure
from bokeh.layouts import column, gridplot
from bokeh.models import CustomJS, ColumnDataSource, Slider, Band, Span
from bokeh.models.layouts import Column
from bokeh.models.ranges import DataRange1d
from bokeh.palettes import Spectral6
import numpy as np
import pandas as pd

from app.utils.model_utils import calculate_dynamic_data

"""
Main plot routines
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
        self.ticker_labels = df_prices.columns
        self.data_len = len(df_prices)
        # Construct data container
        self._generate_data_container(df_prices)
        # Gather plot components
        self._construct_slider()
        self._construct_price_plot()
        # residue plot x range linked with prices plot
        self._construct_residue_plot(x_axis_link=self.price_plot.x_range)
        # Construct layout
        self.p_layout = gridplot([[self.slider_pair_fac],
                                  [column(self.price_plot, self.residue_plot)]])

    def _generate_data_container(self, df_prices: pd.DataFrame) -> None:
        """
        Generates the data required for plotting
        :param df_prices:
        :return:
        """
        df = pd.DataFrame(index=df_prices.index)
        df['y0'] = df_prices.iloc[:, 0]
        df['y1_unscaled'] = df_prices.iloc[:, 1]
        df['x_data'] = datetime(df_prices.index)
        df['x_zeros'] = 0.
        df['scale_factor'] = self.INITIAL_SLIDER_VALUE
        self.data_container = ColumnDataSource(data=df)
        calculate_dynamic_data(self.data_container.data,  self.PRICE_DELTA_MA_WINDOW_DAYS)

    def _construct_slider(self) -> None:
        """
        Constructs slider for pairs multiplier factor
        :return:
        """
        # TODO: intelligent calculation of initial slider parameters
        slider_params = {'start': 0.1*self.INITIAL_SLIDER_VALUE,
                         'end': 10.*self.INITIAL_SLIDER_VALUE,
                         'value': self.INITIAL_SLIDER_VALUE}
        s_step = (slider_params['start'] - slider_params['end']) / 50
        self.slider_pair_fac = Slider(**slider_params, step=s_step, title="Pairs Price Factor")

        def pair_factor_callback(attr, old, new):
            self.data_container.data['scale_factor'] = np.array([new] * self.data_len)
            calculate_dynamic_data(data=self.data_container.data,
                                   ma_window_days=self.PRICE_DELTA_MA_WINDOW_DAYS)

        self.slider_pair_fac.on_change('value', pair_factor_callback)

    def _construct_price_plot(self):
        """
        Constructs price plot (target of slider)
        :return:
        """
        self.price_plot = figure(x_axis_type="datetime", title="Stock Closing Prices",
                                 **self.plot_options)
        # Plot prices
        line0 = self.price_plot.line("x_data", "y0",
                                     source=self.data_container,
                                     muted_alpha=0.2,
                                     color=self.COLORS[0],
                                     legend_label=self.ticker_labels[0])
        line1 = self.price_plot.line("x_data", "y1_times_f",
                                     muted_alpha=0.2,
                                     source=self.data_container,
                                     color=self.COLORS[1],
                                     legend_label=self.ticker_labels[1])

        self.price_plot.grid.grid_line_alpha = 0.3
        self.price_plot.xaxis.axis_label = 'Date'
        self.price_plot.yaxis.axis_label = 'Price'
        self.price_plot.legend.location = "top_left"
        self.price_plot.legend.click_policy= "mute"

    def _construct_residue_plot(self,
                                x_axis_link: DataRange1d):
        """
        Constructs residue plot (target of slider)
        :param x_axis_link: x axis object to couple this plot with
        :return:
        """
        self.residue_plot = figure(x_axis_type="datetime", title="Pair Price Delta",
                                   x_range=x_axis_link,
                                   **self.plot_options)

        # horizontal line
        hline = Span(location=0, dimension='width', line_color='black', line_width=1)
        self.residue_plot.renderers.extend([hline])

        # moving average of residue series
        self.residue_plot.line("x_data", "y_residue_ma",
                               source=self.data_container,
                               color='gray',
                               muted_alpha=0.2,
                               alpha=0.5,
                               line_width=1.5,
                               legend_label=f'{self.PRICE_DELTA_MA_WINDOW_DAYS}D MA')

        # residue series
        self.residue_plot.line("x_data", "y_residue",
                               source=self.data_container,
                               color=self.COLORS[0],
                               legend_label='Price delta')
        band = Band(base='x_data', lower='x_zeros', upper='y_residue', source=self.data_container, level='underlay',
                    fill_alpha=0.2, fill_color='#55FF88')
        self.residue_plot.add_layout(band)

        self.residue_plot.grid.grid_line_alpha = 0.3
        self.residue_plot.xaxis.axis_label = 'Date'
        self.residue_plot.yaxis.axis_label = r'$$\Delta \mathrm{Price}$$'
        self.residue_plot.legend.location = 'bottom_left'

    def get_layout(self) -> Column:
        """
        Returns the bokeh Column layout
        :return:
        """
        return self.p_layout
