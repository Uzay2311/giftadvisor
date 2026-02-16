"""
Gift Advisor - Standalone Flask app for gift recommendations.
Uses havanora-shopify patterns as baseline, focused on occasion-based gift advice.
"""
import logging

from flask import Flask, send_from_directory
from flask_cors import CORS
from gift_advisor import gift_advisor_chat

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

app = Flask(__name__, static_folder="static", static_url_path="")
app.config["DEBUG"] = True
CORS(app, supports_credentials=False)


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(app.static_folder, path)


@app.route("/gift_advisor", methods=["POST", "OPTIONS"])
def gift_advisor_route():
    return gift_advisor_chat()


if __name__ == "__main__":
    app.run()
