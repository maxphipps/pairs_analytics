from threading import Thread
from tornado.ioloop import IOLoop

from app import app
from bokeh.server.server import Server
from bokeh.themes import Theme

from app.plots import Dashboard
from app.utils.file_utils import load_data

from os.path import dirname, join

# To deploy on Heroku:
# FLASK + BOKEH + HEROKU =>
#   https://pjandir.github.io/Bokeh-Heroku-Tutorial/


def bkapp(doc):
    # TODO: Add data pull utility via yfinance/other API

    tickers = ('RDSA.L', 'BP.L')
    # tickers = ('NG.L', 'SSE.L')

    ticker1_ser = load_data(join(dirname(__file__), 'app/data', f'{tickers[0]}.csv')).rename(tickers[0])
    ticker2_ser = load_data(join(dirname(__file__), 'app/data', f'{tickers[1]}.csv')).rename(tickers[1])
    df_prices = ticker1_ser.to_frame().join(ticker2_ser.to_frame(), how='inner')
    df_prices = df_prices.reset_index()

    plot_dashboard_obj = Dashboard(df_prices, ticker_labels=[ticker1_ser.name, ticker2_ser.name]).get_layout()

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
