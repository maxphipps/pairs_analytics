from flask import render_template
from bokeh.embed import server_document
from app import app

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

@app.route('/', methods=['GET'])
def bkapp_page():
    script = server_document('http://localhost:5006/bkapp')
    return render_template("index.html", script=script, template="Flask")
