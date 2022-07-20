from bokeh.plotting import figure
from bokeh.layouts import column, row, gridplot
from bokeh.models import Button, ColumnDataSource, Slider, Band, Span, HoverTool, LinearAxis, Range1d
from bokeh.models.layouts import Column
from bokeh.models.ranges import DataRange1d
from bokeh.palettes import Spectral6
from bokeh.events import DoubleTap
import numpy as np
import pandas as pd

from app.utils.model_utils import calculate_dynamic_data, scan_discontinuities, optimise_scale_factor

"""
Main plot routines
"""


def datetime(x):
    return np.array(x, dtype=np.datetime64)


class Dashboard:
    PRICE_DELTA_MA_WINDOW_DAYS = 30 * 6
    COLORS = Spectral6

    def __init__(self, df_prices: pd.DataFrame, ticker_labels: tuple):
        """
        Dashboard generator
        :param df_prices: Dataframe of price data, indexed by ticker label
        """
        self.plot_options = dict(width=500, plot_height=300, tools='pan,wheel_zoom')
        self.ticker_labels = ticker_labels
        self.data_len = len(df_prices)
        self.discontinuity_idx_ser = pd.Series(dtype=int)
        # Construct data container
        self._generate_data_container(df_prices)
        # Gather plot components
        self._construct_slider()
        self._construct_discontinuity_button()
        self._construct_price_plot()
        self._init_aux_handles()
        # residue plot x range linked with prices plot
        self._construct_residue_plot(x_axis_link=self.price_plot.x_range)
        # Construct layout
        self.p_layout = gridplot([[row(self.slider_pair_fac, self.discontinuity_button)],
                                  [column(self.price_plot, self.residue_plot)]])

    def _generate_data_container(self, df_prices: pd.DataFrame) -> None:
        """
        Generates the data required for plotting
        :param df_prices:
        :return:
        """
        df = pd.DataFrame(index=df_prices.index)
        df['y0'] = df_prices.loc[:, self.ticker_labels[0]]
        df['y1_unscaled'] = df_prices.loc[:, self.ticker_labels[1]]
        df['x_data'] = datetime(df_prices['Date'])
        df['x_zeros'] = 0.
        self.initial_slider_value = df['y0'].mean() / df['y1_unscaled'].mean()
        df['scale_factor'] = self.initial_slider_value
        self.data_container = ColumnDataSource(data=df)
        calculate_dynamic_data(self.data_container.data, self.PRICE_DELTA_MA_WINDOW_DAYS)

    def _init_aux_handles(self):
        """
        Initialises handles to objects
        :return:
        """
        self.optimisation_line = None

    def _construct_slider(self) -> None:
        """
        Constructs slider for pairs multiplier factor
        :return:
        """
        # TODO: intelligent calculation of initial slider parameters
        slider_params = {'start': 0.5*self.initial_slider_value,
                         'end': 2.*self.initial_slider_value,
                         'value': self.initial_slider_value}
        s_step = (slider_params['start'] - slider_params['end']) / 50
        self.slider_pair_fac = Slider(**slider_params, step=s_step, title="Pairs Price Factor")

        def pair_factor_callback(attr, old, new):
            self.data_container.data['scale_factor'] = np.array([new] * self.data_len)
            calculate_dynamic_data(data=self.data_container.data,
                                   ma_window_days=self.PRICE_DELTA_MA_WINDOW_DAYS)

        self.slider_pair_fac.on_change('value', pair_factor_callback)

    def _construct_discontinuity_button(self) -> None:
        """
        Constructs button to add new discontinuity
        :return:
        """
        def discontinuity_button_callback():
            # Hide moving average during optimisation by overriding with Nones
            self.data_container.data['y_residue_ma'] = [None] * self.data_len

            # Scan space of possible discontinuities
            res = scan_discontinuities(self.data_container.data, self.slider_pair_fac.value, self.discontinuity_idx_ser)

            # Hide old optimisation line
            if self.optimisation_line:
                self.optimisation_line.visible = False

            # Plot results summary on secondary axis
            line_kwargs = {'y_range_name': 'OptimisationScore', 'muted_alpha': 0.2,
                           'source': ColumnDataSource.from_df(res), 'color': 'gray'}
                           # 'legend_label': 'Discontinuity Optimisation Cost'
            self.optimisation_line = self.price_plot.line("x_timestamp", "norm_net_cost", **line_kwargs)

            # Extract optimised values
            # Exclude previously identified discontinuities
            is_new_discontinuity = ~res['discontinuity_idx'].isin(self.discontinuity_idx_ser)
            _idxmin = res['norm_net_cost'][is_new_discontinuity].astype(float).idxmin()
            opt_vals = res.loc[_idxmin]  # idxmin uses loc
            opt_x = opt_vals['x_timestamp']

            # Display optimised discontinuity location
            vline = Span(location=opt_x, dimension='height', line_color='red', line_width=1)
            self.price_plot.renderers.extend([vline])

            # Transform data
            df_data = pd.DataFrame(self.data_container.data)
            df_data = df_data.drop(columns='index')
            discontinuity_idx = opt_vals['discontinuity_idx']
            discontinuity_idx_prev = opt_vals['discontinuity_idx_prev']
            discontinuity_idx_next = opt_vals['discontinuity_idx_next']

            # Add newly identified discontinuity to master list
            self.discontinuity_idx_ser = self.discontinuity_idx_ser.append(pd.Series([discontinuity_idx]))

            df_data.loc[discontinuity_idx_prev:discontinuity_idx, 'scale_factor'] = opt_vals['l_scale_factor']
            df_data.loc[discontinuity_idx:discontinuity_idx_next, 'scale_factor'] = opt_vals['r_scale_factor']

            self.data_container.data = ColumnDataSource.from_df(df_data)

            # Unhide moving average by overriding Nones
            calculate_dynamic_data(data=self.data_container.data,
                                   ma_window_days=self.PRICE_DELTA_MA_WINDOW_DAYS)

        self.discontinuity_button = Button(label="Find Discontinuity")
        self.discontinuity_button.on_click(discontinuity_button_callback)

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
        self.price_plot.add_tools(HoverTool(tooltips=[('x', '@x_data{%F}')],
                                            formatters={'@x_data': 'datetime'},
                                            renderers=[line0], mode="vline"))

        # Add secondary axis
        self.price_plot.extra_y_ranges = {"OptimisationScore": Range1d(0, 100)}
        self.price_plot.add_layout(LinearAxis(y_range_name="OptimisationScore",
                                              axis_label="Optimisation Score"), 'right')

        def callback(event):
            vline = Span(location=event.x, dimension='height', line_color='red', line_width=1)
            self.price_plot.renderers.extend([vline])

        self.price_plot.on_event(DoubleTap, callback)

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
