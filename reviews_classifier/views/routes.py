from flask import Blueprint, render_template

classifier = Blueprint('classifier', __name__, static_folder='static')


@classifier.route('/')
def main():
    """
    Route for welcome page, it's like menu for further interactions
    """
    return render_template("main.html")
