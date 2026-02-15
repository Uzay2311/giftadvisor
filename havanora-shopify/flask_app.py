from flask import Flask, request
from flask_cors import CORS
from chat_bot import chat_bot, widget_config, customize_widget
from insert_chat_telemetry import telemetry_bp
from seal_subs import seal_subs, shopify_webhook
from fetch_customer_details import fetch_customer_details
from new_agent import new_agent
from inbox_api import inbox
from domain_delete import domain_delete  # NEW
from hn_chat_bot import hn_chat  # NEW

# NEW (HN sharded telemetry + reads)
from hn_telemetry import hn_telemetry
from hn_read_customer_threads import hn_read_customer_threads


app = Flask(__name__)
app.config["DEBUG"] = True
# Allow only your dashboard hosts (apex + www for both spellings)
CORS(
    app,
    # resources={r"/*": {"origins": [
    #     "https://workiqapp.com",
    #     "https://www.workiqapp.com",
    #     "https://workiapp.com",
    #     "https://www.workiapp.com",
    # ]}},
    supports_credentials=False,
)

# register blueprint
app.register_blueprint(telemetry_bp)

@app.route("/domain_delete", methods=["POST", "OPTIONS"])   # NEW
def domain_delete_route():
    return domain_delete()

@app.route("/chat_bot", methods=["POST"])
def chat_bot_route():
    return chat_bot()

# Shopify customer portal endpoint (Liquid uses: /apps/havanora/chat)
@app.route("/hn_chat_bot", methods=["POST", "OPTIONS"])  # NEW
def hn_chat_route():
    return hn_chat()

# NEW: sharded telemetry write/touch
@app.route("/hn_telemetry", methods=["POST", "OPTIONS"])
def hn_telemetry_route():
    return hn_telemetry()

# NEW: fetch customer + last 10 threads for left pane
@app.route("/hn_read_customer_threads", methods=["POST", "OPTIONS"])
def hn_read_customer_threads_route():
    return hn_read_customer_threads()

@app.route("/customize_widget", methods=["GET", "POST"])
def customize_widget_route():
    return customize_widget()

@app.route("/inbox", methods=["GET", "POST"])
def inbox_route():
    return inbox()

@app.route("/new_agent", methods=["GET", "POST", "OPTIONS"])
def new_agent_route():
    return new_agent()

@app.route("/fetch_customer_details", methods=["GET", "POST"])
def fetch_customer_details_route():
    return fetch_customer_details()

@app.route("/seal_subs", methods=["POST"])
def seal_subs_route():
    return seal_subs()

@app.route("/shopify_webhook", methods=["POST"])
def shopify_webhook_route():
    return shopify_webhook()

@app.route("/widget_config", methods=["GET", "POST"])
def widget_config_route():
    return widget_config()

if __name__ == "__main__":
    app.run()
