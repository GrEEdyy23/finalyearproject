from flask import Flask, render_template, request,flash, redirect, url_for
from keras.saving.experimental.saving_lib import load_model
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')
if __name__ == '__main__':
    app.run()
