from flask import render_template
from app import app
from bokeh.embed import components

from .plots import gen_dashboard


@app.route('/')
@app.route('/index')
def index():
    """
    Main dashboard page
    :return:
    """
    from bokeh.sampledata.stocks import AAPL, GOOG, IBM, MSFT

    plot_dashboard_obj = gen_dashboard(ticker_tups=(#('AAPL', AAPL),
                                                   #('GOOG',GOOG),
                                                   ('IBM',IBM),
                                                   ('MSFT',MSFT),
                                                   ))

    _script, _div = components(plot_dashboard_obj)
    plot_dashboard = {'script': _script, 'div': _div}

    return render_template('index.html', title='plot',
                           # plot_prices=plot_prices,
                           # plot_ma=plot_ma,
                           plot_dashboard=plot_dashboard)
