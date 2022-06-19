from threading import Thread
from tornado.ioloop import IOLoop

from app import app
from bokeh.server.server import Server
from bokeh.themes import Theme

from bokeh.sampledata.stocks import AAPL, GOOG, IBM, MSFT

from app.plots import Dashboard
from app.utils.file_utils import load_data

from os.path import dirname, join

# To deploy on Heroku:
# FLASK + BOKEH + HEROKU =>
#   https://pjandir.github.io/Bokeh-Heroku-Tutorial/


def bkapp(doc):
    # TODO: Add data pull utility via yfinance/other API
    RDSA = load_data(join(dirname(__file__), 'app/data', 'RDSA.L.csv')).set_index('Date')['Adj Close'].rename('RDSA.L')
    BP = load_data(join(dirname(__file__), 'app/data', 'BP.L.csv')).set_index('Date')['Adj Close'].rename('BP.L')
    df_prices = RDSA.to_frame().join(BP.to_frame(), how='inner')
    plot_dashboard_obj = Dashboard(df_prices).get_plot()

    doc.add_root(plot_dashboard_obj)
    doc.theme = Theme(filename="theme.yaml")


def bk_worker():
    """
    Main dashboard page
    :return:
    """
    # Can't pass num_procs > 1 in this configuration. If you need to run multiple
    # processes, see e.g. flask_gunicorn_embed.py
    server = Server({'/bkapp': bkapp}, io_loop=IOLoop(), allow_websocket_origin=["localhost:8000", "127.0.0.1:8000"])
    server.start()
    server.io_loop.start()


Thread(target=bk_worker).start()


# https://blog.miguelgrinberg.com/post/setting-up-a-flask-application-in-pycharm
# Use different version of run in order to enable the debugger
# "More specifically, this is a bug that affects Flask applications that are started with the flask run command,
# but works well with the old app.run() method of running the application."
if __name__ == '__main__':
    print('Opening single process Flask app with embedded Bokeh application on http://localhost:8000/')
    print()
    print('Multiple connections may block the Bokeh app in this configuration!')
    print('See "flask_gunicorn_embed.py" for one way to run multi-process')
    app.run(debug=True, use_debugger=False, use_reloader=False, passthrough_errors=True, port=8000)
