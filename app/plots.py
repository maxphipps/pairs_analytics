from bokeh.plotting import figure
from bokeh.layouts import column, row, gridplot
from bokeh.events import DoubleTap
from bokeh.models import Button, ColumnDataSource, Slider, DateSlider, Band, Span, HoverTool, Panel, Tabs, CheckboxGroup
from bokeh.models.layouts import Column
from bokeh.models.ranges import DataRange1d
from bokeh.palettes import Spectral6, Spectral10
import numpy as np
import pandas as pd
import datetime

from app.utils.constants import PRICE_DELTA_MA_WINDOW_DAYS
from app.utils.model_utils import calculate_dynamic_data, optimise_hedge_ratio

"""
Main plot routines
"""


def to_datetime_array(x):
    return np.array(x, dtype=np.datetime64)

class Dashboard:
    COLORS = Spectral6
    NUM_FUNCTIONS = 3

    def __init__(self, df_prices: pd.DataFrame, ticker_labels: tuple):
        """
        Dashboard generator
        :param df_prices: Dataframe of price data, indexed by ticker label
        """
        self.plot_options = dict(width=500, plot_height=300, tools='pan,wheel_zoom')
        self.ticker_labels = ticker_labels
        self.data_length = len(df_prices)
        # Construct data container
        self._init_data_container(df_prices)
        self.x_axis_minimum = datetime.datetime.fromtimestamp(float(self.data_container.data['x_data'][0]) * 1E-9)
        self.x_axis_maximum = datetime.datetime.fromtimestamp(float(self.data_container.data['x_data'][-1]) * 1E-9)
        # Gather plot components
        self._construct_hedge_ratio_slider()
        self._construct_discontinuity_button()
        self._construct_price_plot()
        self._construct_hedge_ratio_plot(x_axis_link=self.price_plot.x_range)
        self._construct_hedge_ratio_widgets()
        self._initialise_hedging_function_data()
        self._construct_spread_plot(x_axis_link=self.price_plot.x_range)
        # Construct layout
        self.p_layout = gridplot([[self.price_plot, column([self.discontinuity_button, self.hedge_ratio_widget_tabs])],
                                  [self.hedge_ratio_plot, self.hedge_ratio_slider],
                                  [self.spread_plot]],
                                 width=400, height=250)

    def plot_x_to_model_x(self, plot_x_value):
        ratio_of_x_axis = (plot_x_value - self.x_axis_minimum) / (self.x_axis_maximum - self.x_axis_minimum)
        x_clicked_in_data_units = self.data_length * ratio_of_x_axis
        return x_clicked_in_data_units

    def _construct_hedge_ratio_widgets(self):
        def initial_midpoint_slider_callback(attr, old, new):
            selected_function_idx = self.hedge_ratio_widget_tabs.active
            if isinstance(new, np.datetime64):
                # TODO: below conversion of numpy to datetime seems a little hacky
                _datetime = datetime.datetime.fromtimestamp(new.astype(datetime.datetime) * 1E-9)
                self.mdl_params[selected_function_idx]['x0'] = self.plot_x_to_model_x(_datetime)
            else:
                self.mdl_params[selected_function_idx]['x0'] = new

        def function_is_active_checkbox_callback(event):
            selected_function_idx = self.hedge_ratio_widget_tabs.active
            # If active
            if self.function_is_active_checkbox[selected_function_idx].active == [0]:
                self.mdl_params[selected_function_idx]['active'] = 1
            else:
                self.mdl_params[selected_function_idx]['active'] = 0

        # Tab elements
        self.function_is_active_checkbox = []
        self.function_initial_midpoint_slider = []
        # Compile into tabs
        tabs = []
        for itab in range(self.NUM_FUNCTIONS):
            self.function_is_active_checkbox.append(CheckboxGroup(labels=['Active'], active=[]))
            self.function_is_active_checkbox[-1].on_click(function_is_active_checkbox_callback)

            self.function_initial_midpoint_slider.append(DateSlider(start=self.x_axis_minimum.date(),
                                                                    end=self.x_axis_maximum.date(),
                                                                    value=self.x_axis_minimum.date(),
                                                                    disabled=True,
                                                                    step=1, title="Midpoint initial value"))
            self.function_initial_midpoint_slider[-1].on_change('value', initial_midpoint_slider_callback)

            tabs.append(Panel(child=gridplot([[self.function_is_active_checkbox[itab]],
                                              [self.function_initial_midpoint_slider[itab]]]),
                              title=str(itab+1)))
        self.hedge_ratio_widget_tabs = Tabs(tabs=tabs)

    def _initialise_hedging_function_data(self):
        """
        Initialise all hedging functions and update widgets accordingly.
        Only first hedging function is active.
        :return:
        """
        init_x0 = int(self.data_length/2)
        self.mdl_params = [{'l': 0.0, 'm': 0.0, 'k': 0.01, 'x0': init_x0, 'active': 0}
                           for ii in range(self.NUM_FUNCTIONS)]
        self.mdl_params[0]['l'] = self.initial_slider_value
        self.mdl_params[0]['m'] = self.initial_slider_value
        self.mdl_params[0]['active'] = 1

        # Update 1st function's tab data
        self.function_is_active_checkbox[0].active = [0]
        self.function_initial_midpoint_slider[0].value = self.data_container.data['x_data'][init_x0]

        calculate_dynamic_data(self.data_container.data, self.mdl_params)

    def _init_data_container(self, df_prices: pd.DataFrame) -> None:
        """
        Generates the data required for plotting
        :param df_prices:
        :return:
        """
        df = pd.DataFrame()
        df['y0'] = df_prices.loc[:, self.ticker_labels[0]]
        df['y1_unscaled'] = df_prices.loc[:, self.ticker_labels[1]]
        df['x_data'] = to_datetime_array(df_prices['Date'])
        df['x_zeros'] = 0.
        df['x_index'] = df.index.copy(deep=True)
        self.data_container = ColumnDataSource(data=df)

    def _construct_hedge_ratio_slider(self) -> None:
        """
        Constructs slider for hedge ratio
        :return:
        """
        self.initial_slider_value = self.data_container.data['y0'].mean() / self.data_container.data['y1_unscaled'].mean()
        slider_params = {'start': 0.5*self.initial_slider_value,
                         'end': 2.*self.initial_slider_value,
                         'value': self.initial_slider_value,
                         'orientation': 'vertical',
                         'direction': 'rtl'}
        s_step = (slider_params['start'] - slider_params['end']) / 50
        self.hedge_ratio_slider = Slider(**slider_params, step=s_step, title="Pairs Price Factor")

        def pair_factor_callback(attr, old, new):
            self.mdl_params[0]['l'] = new
            self.mdl_params[0]['m'] = new
            calculate_dynamic_data(self.data_container.data, self.mdl_params)

        self.hedge_ratio_slider.on_change('value', pair_factor_callback)

    def _construct_discontinuity_button(self) -> None:
        """
        Constructs button to add new discontinuity
        :return:
        """
        def discontinuity_button_callback():
            # Hide moving average during optimisation by overriding with Nones
            self.data_container.data['y_spread_ma'] = [None] * self.data_length

            # Optimise
            self.discontinuity_button.label = "Optimising..."
            opt_params = optimise_hedge_ratio(self.data_container.data, self.mdl_params, num_functions=self.NUM_FUNCTIONS)
            # Update params
            for ifunc in range(self.NUM_FUNCTIONS):
                if self.mdl_params[ifunc].get('active'):
                    for param in ('l', 'm', 'k', 'x0'):
                        self.mdl_params[ifunc][param] = opt_params[ifunc][param]
            self.discontinuity_button.label = "Find Discontinuity"

            # # TODO: Display optimised discontinuity location
            # # TODO: Interval instead of single line
            # vline = Span(location=self.mdl_params['x0'], dimension='height', line_color='red', line_width=1)
            # self.price_plot.renderers.extend([vline])

            # Unhide moving average by overriding Nones
            calculate_dynamic_data(self.data_container.data, self.mdl_params)

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
        line0 = self.price_plot.line("x_data", "y0", source=self.data_container, muted_alpha=0.2,
                                     color=self.COLORS[0], legend_label=self.ticker_labels[0])
        line1 = self.price_plot.line("x_data", "y1_times_f", muted_alpha=0.2, source=self.data_container,
                                     color=self.COLORS[1], legend_label=self.ticker_labels[1])

        self.price_plot.grid.grid_line_alpha = 0.3
        self.price_plot.xaxis.axis_label = 'Date'
        self.price_plot.yaxis.axis_label = 'Price'
        self.price_plot.legend.location = "top_left"
        self.price_plot.legend.click_policy= "mute"
        self.price_plot.add_tools(HoverTool(tooltips=[('x', '@x_data{%F}')],
                                            formatters={'@x_data': 'datetime'},
                                            renderers=[line0], mode="vline"))
        # 1 vline per logistic function, hidden
        self.vlines = []
        for ivline in range(self.NUM_FUNCTIONS):
            self.vlines.append(Span(location=0, dimension='height', line_color=Spectral10[ivline],
                                    line_width=1, visible=False))
            self.price_plot.add_layout(self.vlines[-1])

        def price_plot_callback(event):
            # Show line
            selected_function_idx = self.hedge_ratio_widget_tabs.active
            self.vlines[selected_function_idx].visible = True
            self.vlines[selected_function_idx].location = event.x
            self.function_is_active_checkbox[selected_function_idx].active = [0]

            # Update slider with new initial midpoint value
            selected_function_idx = self.hedge_ratio_widget_tabs.active
            self.function_initial_midpoint_slider[selected_function_idx].value = event.x

            # Set function x0 value
            datetime_clicked = datetime.datetime.fromtimestamp(event.x * 1E-3)
            x_clicked_in_data_units = self.plot_x_to_model_x(datetime_clicked)
            self.mdl_params[selected_function_idx]['x0'] = x_clicked_in_data_units

        self.price_plot.on_event(DoubleTap, price_plot_callback)

    def _construct_hedge_ratio_plot(self, x_axis_link: DataRange1d):
        """
        Constructs hedge ratio plot
        :param x_axis_link: x axis object to couple this plot with
        :return:
        """
        self.hedge_ratio_plot = figure(x_axis_type="datetime", title="Hedge Ratio", x_range=x_axis_link,
                                       **self.plot_options)
        line = self.hedge_ratio_plot.line("x_data", "hedge_ratio", source=self.data_container, color=self.COLORS[0])

        self.hedge_ratio_plot.grid.grid_line_alpha = 0.3
        self.hedge_ratio_plot.xaxis.axis_label = 'Date'
        self.hedge_ratio_plot.yaxis.axis_label = 'Weight'
        tooltips = [(self.ticker_labels[0], '1.00'),
                    (self.ticker_labels[1], '@hedge_ratio{(0.00)}')]
        self.hedge_ratio_plot.add_tools(HoverTool(tooltips=tooltips,
                                                  formatters={'@hedge_ratio': 'numeral'},
                                                  renderers=[line], mode="vline"))

    def _construct_spread_plot(self, x_axis_link: DataRange1d):
        """
        Constructs spread plot (target of slider)
        :param x_axis_link: x axis object to couple this plot with
        :return:
        """
        self.spread_plot = figure(x_axis_type="datetime", title="Pair Price Delta", x_range=x_axis_link,
                                  **self.plot_options)

        # horizontal line
        hline = Span(location=0, dimension='width', line_color='black', line_width=1)
        self.spread_plot.renderers.extend([hline])

        # moving average of spread series
        self.spread_plot.line("x_data", "y_spread_ma", source=self.data_container, color='gray', muted_alpha=0.2,
                              alpha=0.5, line_width=1.5, legend_label=f'{PRICE_DELTA_MA_WINDOW_DAYS}D MA')

        # spread series
        self.spread_plot.line("x_data", "y_spread", source=self.data_container, color=self.COLORS[0],
                              legend_label='Price delta')
        band = Band(base='x_data', lower='x_zeros', upper='y_spread', source=self.data_container, level='underlay',
                    fill_alpha=0.2, fill_color='#55FF88')
        self.spread_plot.add_layout(band)

        self.spread_plot.grid.grid_line_alpha = 0.3
        self.spread_plot.xaxis.axis_label = 'Date'
        self.spread_plot.yaxis.axis_label = r'$$\Delta \mathrm{Price}$$'
        self.spread_plot.legend.location = 'bottom_left'

    def get_layout(self) -> Column:
        """
        Returns the bokeh Column layout
        :return:
        """
        return self.p_layout
