from flask import Flask, render_template
import sys
import logging

app = Flask(__name__)
app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)

@app.route('/')
def constellano():
    return "Hello World!"
