
from app import app

# To deploy on Heroku:
# FLASK + BOKEH + HEROKU =>
#   https://pjandir.github.io/Bokeh-Heroku-Tutorial/

# https://blog.miguelgrinberg.com/post/setting-up-a-flask-application-in-pycharm
# Use different version of run in order to enable the debugger
# "More specifically, this is a bug that affects Flask applications that are started with the flask run command,
# but works well with the old app.run() method of running the application."
if __name__ == '__main__':
    app.run(debug=True, use_debugger=False, use_reloader=False, passthrough_errors=True)