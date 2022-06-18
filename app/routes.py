from flask import render_template
from app import app
from bokeh.embed import components

from .plots import Dashboard


@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


@app.route('/')
@app.route('/index')
def index():
    """
    Main dashboard page
    :return:
    """
    from bokeh.sampledata.stocks import AAPL, GOOG, IBM, MSFT

    plot_dashboard_obj = Dashboard(ticker_tuples=(#('AAPL', AAPL),
                                                   #('GOOG',GOOG),
                                                   ('IBM',IBM),
                                                   ('MSFT',MSFT),
                                                   )).get_plot()

    _script, _div = components(plot_dashboard_obj)
    plot_dashboard = {'script': _script, 'div': _div}

    return render_template('index.html', title='plot',
                           # prices_plot=prices_plot,
                           # plot_ma=plot_ma,
                           plot_dashboard=plot_dashboard)
