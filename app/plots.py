from bokeh.plotting import figure
import numpy as np

from utils.math_utils import rolling_mean


def datetime(x):
    return np.array(x, dtype=np.datetime64)


def gen_dashboard(ticker_tups):
    from bokeh.layouts import gridplot
    from bokeh.palettes import Spectral6

    plot_options = dict(width=500, plot_height=300, tools='pan,wheel_zoom')

    # todo: palette
    # colors = ('#A6CEE3','#B2DF8A','#33A02C','#FB9A99')
    colors = Spectral6

    # unzip ticker_tups
    all_ticker_name, all_ticker_data = zip(*ticker_tups)

    # initial slider value setting
    s_value = 3.66

    _init_y0 = all_ticker_data[0]['adj_close']
    _init_y1 = all_ticker_data[1]['adj_close']

    # initial values, for dynamic components
    # _yzeros = np.zeros(all_ticker_data[1]['date'])
    _init_y1_times_f = np.multiply(_init_y1, s_value)
    _init_y_residue = _init_y1_times_f - _init_y0

    # Window size for price delta moving average
    window_size = 30*6

    # x axis values for price residue plot
    x_residue = datetime(all_ticker_data[0]['date'])

    # container for prices data
    # data_dict = {}
    # for ticker_tup in ticker_tups:
    #     ticker_name, ticker_data = ticker_tup
    #     data_dict[ticker_name] = ticker_data
    # the x and y for plotting tickers 1 and 2
    data_dict_prices = dict(x0=datetime(all_ticker_data[0]['date']),
                            y0=_init_y0,
                            x1=datetime(all_ticker_data[1]['date']),
                            y1=_init_y1,
                            y1_times_f=_init_y1_times_f,
                            x_residue=x_residue,
                            y_residue=_init_y_residue,
                            # x_residue_ma=x_residue[window_size:],
                            x_residue_ma=x_residue,
                            y_residue_ma=rolling_mean(_init_y_residue, window_size),
                            )


    '''
    Slider for pairs multiplier factor
    '''

    from bokeh.layouts import column
    from bokeh.models import CustomJS, ColumnDataSource, Slider, Band

    # data container to ColumnDataSource
    source_pairFac = ColumnDataSource(data=data_dict_prices)

    # TODO: intelligent calculation of initial slider parameters
    s_start = 0.1
    s_end = 10.
    s_step = (s_end - s_start) / 50  # .1
    slider_pairFac = Slider(start=s_start, end=s_end, value=s_value, step=s_step, title="Pairs Price Factor")

    # Slider: updates y1_times_f = y1 * factor
    update_price_curve = CustomJS(args=dict(source_pairFac=source_pairFac, slider_pairFac=slider_pairFac), code="""
        var data = source_pairFac.data;
        var f = slider_pairFac.value;
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
            
            // update the residue values moving average
            if (i >= ndays_ma) {
                i_ma = i-ndays_ma;
                y_residue_ma[i_ma] = y_residue[i];
                // cumsum the previous n days
                //for (var j = i_ma; j > Math.max(0,i_ma-ndays_ma); j--) {
                for (var j = i; j > Math.max(0,i-ndays_ma); j--) {
                    y_residue_ma[i_ma] += y_residue[j];
                }
                y_residue_ma[i_ma] /= ndays_ma;
            }
            
        }

        // necessary because we mutated source_pairFac.data in-place
        source_pairFac.change.emit();
    """)
    slider_pairFac.js_on_change('value', update_price_curve)


    '''
    Plot prices (is target of slider)
    '''

    plot_prices = figure(x_axis_type="datetime", title="Stock Closing Prices",
                         **plot_options)

    # plot each ticker
    # for iticker, ticker_tup in enumerate(ticker_tups):
    #     ticker_name, ticker_data = ticker_tup
    #     plot_prices.line(datetime(ticker_data['date']),
    #                      ticker_data['adj_close'],
    #                      color=colors[iticker],
    #                      legend_label=ticker_name)

    # 1st ticker
    plot_prices.line("x0", "y0",
                     source=source_pairFac,
                     color=colors[0],
                     legend_label=all_ticker_name[0])
    # 2nd ticker
    plot_prices.line("x1", "y1_times_f",
                     source=source_pairFac,
                     color=colors[1],
                     legend_label=all_ticker_name[1])

    plot_prices.grid.grid_line_alpha = 0.3
    plot_prices.xaxis.axis_label = 'Date'
    plot_prices.yaxis.axis_label = 'Price'
    plot_prices.legend.location = "top_left"


    '''
    Plot residue (is target of slider)
    '''

    # Linked x range with plot_prices
    plot_residue = figure(x_axis_type="datetime", title="Pair Price Delta",
                          x_range = plot_prices.x_range,
                          **plot_options)

    # Horizontal line
    from bokeh.models import Span
    hline = Span(location=0, dimension='width', line_color='black', line_width=1)
    plot_residue.renderers.extend([hline])

    # the residue data moving average
    plot_residue.line("x_residue_ma", "y_residue_ma",
                      source=source_pairFac,
                      color='gray',
                      alpha=0.5,
                      line_width=1.5,
                      legend_label=f'{window_size}D MA',
                      )

    # the residue data
    plot_residue.line("x_residue", "y_residue",
                      source=source_pairFac,
                      color=colors[0],
                      legend_label='Price Delta',
                      )

    #
    band = Band(base='x_residue', upper='y_residue', source=source_pairFac, level='underlay',
                fill_alpha=0.2, fill_color='#55FF88')
    plot_residue.add_layout(band)

    plot_residue.grid.grid_line_alpha = 0.3
    plot_residue.xaxis.axis_label = 'Date'
    plot_residue.yaxis.axis_label = 'Price'
    plot_residue.legend.location = "top_left"

    # '''
    # Plot moving average
    # '''
    #
    # # Linked range with plot_prices
    # plot_ma = figure(x_axis_type="datetime", title="One-Month Average Adj. Close",
    #                  x_range=plot_prices.x_range, y_range=plot_prices.y_range,
    #                  **plot_options)
    #
    # for iticker, ticker_tup in enumerate(ticker_tups):
    #     ticker_name, ticker_data = ticker_tup
    #     ticker_prices = np.array(ticker_data['adj_close'])
    #     ticker_dates = np.array(ticker_data['date'], dtype=np.datetime64)
    #
    #     window_size = 30
    #     ticker_avg = running_mean(ticker_prices, window_size)
    #
    #     # plot_ma.circle(ticker_dates, ticker_prices, size=4, legend_label='close',
    #     #                color='darkgrey', alpha=0.2)
    #
    #     plot_ma.line(ticker_dates, ticker_avg,
    #                  color=colors[iticker],
    #                  legend_label=ticker_name)
    #
    # plot_ma.grid.grid_line_alpha = 0.3
    # plot_ma.xaxis.axis_label = 'Date'
    # plot_ma.yaxis.axis_label = 'Price'
    # plot_ma.legend.location = "top_left"

    p = gridplot([[slider_pairFac],
                  [column(plot_prices,plot_residue)]])
    # p = gridplot([[column(slider_pairFac, plot_prices), plot_ma])

    # show the results
    # show(p)

    return p
