import os
import requests
import time
from datetime import datetime
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS  # <-- added for CORS support
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# ---------- CONFIGURATION ----------
ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Debug: print keys (masked)
print(f"🔑 ETHERSCAN_API_KEY: {'✅ set' if ETHERSCAN_API_KEY else '❌ missing'}")
print(f"🔑 OPENAI_API_KEY: {'✅ set' if OPENAI_API_KEY else '❌ missing'}")
print("🌐 CORS enabled for all routes")

if not ETHERSCAN_API_KEY:
    print("⚠️  WARNING: ETHERSCAN_API_KEY not set in .env")
if not OPENAI_API_KEY:
    print("⚠️  WARNING: OPENAI_API_KEY not set – AI will NOT work!")

ETHERSCAN_BASE = "https://api.etherscan.io/v2/api"
COINGECKO_PRICE_URL = "https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd"

# Initialize OpenAI client ONLY if key is present
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ---------- Helper: fetch ETH price ----------
def get_eth_usd():
    try:
        resp = requests.get(COINGECKO_PRICE_URL, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return data["ethereum"]["usd"]
    except:
        return None

# ---------- Helper: fetch block timestamp ----------
def get_block_timestamp(block_number):
    try:
        params = {
            "module": "proxy",
            "action": "eth_getBlockByNumber",
            "blocknumber": hex(block_number),
            "apikey": ETHERSCAN_API_KEY,
            "chainid": 1
        }
        resp = requests.get(ETHERSCAN_BASE, params=params)
        resp.raise_for_status()
        data = resp.json()
        if "result" in data and data["result"]:
            timestamp_hex = data["result"].get("timestamp")
            if timestamp_hex:
                if isinstance(timestamp_hex, int):
                    return timestamp_hex
                if isinstance(timestamp_hex, str):
                    if timestamp_hex.startswith("0x"):
                        return int(timestamp_hex, 16)
                    else:
                        return int(timestamp_hex)
    except:
        pass
    return None

# ---------- Risk scoring ----------
def calculate_risk(tx_data):
    score = 0
    reasons = []
    value_eth = int(tx_data.get("value", "0x0"), 16) / 10**18

    if value_eth > 100:
        score += 30
        reasons.append("Large transfer (>100 ETH)")
    elif value_eth > 10:
        score += 15
        reasons.append("Moderate transfer (>10 ETH)")

    to_addr = tx_data.get("to", "").lower()
    from_addr = tx_data.get("from", "").lower()
    if to_addr == "0x0000000000000000000000000000000000000000":
        score += 20
        reasons.append("Sent to zero address (burn)")
    if from_addr == "0x0000000000000000000000000000000000000000":
        score += 20
        reasons.append("Sent from zero address (mint)")

    input_data = tx_data.get("input", "0x")
    if input_data and input_data != "0x":
        score += 10
        reasons.append("Contract interaction (calldata)")

    logs = tx_data.get("receipt", {}).get("logs", [])
    if len(logs) > 5:
        score += 10
        reasons.append("Many events logged")

    score = min(score, 100)
    if score < 30:
        level = "Low"
    elif score < 60:
        level = "Medium"
    else:
        level = "High"
    return score, level, reasons

# ---------- Etherscan fetcher ----------
def fetch_transaction(tx_hash):
    tx_params = {
        "module": "proxy",
        "action": "eth_getTransactionByHash",
        "txhash": tx_hash,
        "apikey": ETHERSCAN_API_KEY,
        "chainid": 1
    }
    tx_resp = requests.get(ETHERSCAN_BASE, params=tx_params)
    tx_resp.raise_for_status()
    tx_data = tx_resp.json()
    if "error" in tx_data:
        raise Exception(f"Etherscan error: {tx_data['error']}")
    tx = tx_data.get("result")
    if not tx or not isinstance(tx, dict):
        raise Exception("Transaction not found or invalid response.")

    receipt_params = {
        "module": "proxy",
        "action": "eth_getTransactionReceipt",
        "txhash": tx_hash,
        "apikey": ETHERSCAN_API_KEY,
        "chainid": 1
    }
    receipt_resp = requests.get(ETHERSCAN_BASE, params=receipt_params)
    receipt_resp.raise_for_status()
    receipt_data = receipt_resp.json()
    if "error" in receipt_data:
        raise Exception(f"Etherscan receipt error: {receipt_data['error']}")
    receipt = receipt_data.get("result")
    if not receipt or not isinstance(receipt, dict):
        raise Exception("Receipt not found.")
    tx["receipt"] = receipt

    block_num = int(tx.get("blockNumber", "0x0"), 16)
    timestamp = get_block_timestamp(block_num)
    if timestamp is not None:
        tx["blockTimestamp"] = timestamp

    return tx

# ---------- Build prompt ----------
def build_prompt(tx_data):
    tx = tx_data
    receipt = tx.get("receipt", {})
    value_eth = int(tx.get("value", "0x0"), 16) / 10**18
    gas_used = int(receipt.get("gasUsed", "0x0"), 16)
    gas_price = int(tx.get("gasPrice", "0x0"), 16)
    gas_fee_eth = (gas_used * gas_price) / 10**18
    status = "Success" if receipt.get("status") == "0x1" else "Failed"

    prompt = f"""
You are a blockchain transaction explainer. Given the transaction data below, provide a **very short summary** (3‑5 sentences) covering:
- What type of transaction (transfer, swap, contract call, etc.)
- Which assets and how much were moved
- The gas fee and whether it's normal
- Any unusual or notable behavior

Transaction details:
- From: {tx.get('from')}
- To: {tx.get('to')}
- Value: {value_eth:.6f} ETH
- Gas used: {gas_used}
- Gas price: {gas_price} wei
- Gas fee: {gas_fee_eth:.6f} ETH
- Input data (hex): {tx.get('input', '0x')}
- Status: {status}
- Logs: {receipt.get('logs', [])}

Keep it brief – no extra details.
"""
    return prompt

# ---------- AI explanation using gpt-4o-mini (ONLY) ----------
def get_ai_explanation(prompt):
    if not openai_client:
        return "❌ OpenAI API key is missing or invalid. Please set OPENAI_API_KEY in .env"
    try:
        print("🤖 Using OpenAI gpt-4o-mini...")
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a blockchain expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=150
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ OpenAI error: {str(e)}"

# ---------- Flask routes ----------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/explain", methods=["POST", "OPTIONS"])
def explain():
    # Handle preflight OPTIONS request
    if request.method == "OPTIONS":
        response = jsonify({"message": "OK"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
        response.headers.add("Access-Control-Allow-Methods", "POST")
        return response, 200

    # Actual POST handling
    tx_hash = request.form.get("tx_hash", "").strip()
    if not tx_hash:
        return jsonify({"error": "Transaction hash is required."}), 400

    try:
        tx_data = fetch_transaction(tx_hash)
        risk_score, risk_level, risk_reasons = calculate_risk(tx_data)

        eth_price = get_eth_usd()
        gas_fee_eth = (int(tx_data["receipt"].get("gasUsed", "0x0"), 16) *
                       int(tx_data.get("gasPrice", "0x0"), 16)) / 10**18
        gas_fee_usd = gas_fee_eth * eth_price if eth_price else None

        timestamp = tx_data.get("blockTimestamp")
        if timestamp is not None:
            if isinstance(timestamp, str):
                try:
                    timestamp = int(timestamp, 16) if timestamp.startswith("0x") else int(timestamp)
                except:
                    timestamp = None

        if timestamp:
            age_seconds = int(time.time() - timestamp)
            if age_seconds < 60:
                age_str = f"{age_seconds} seconds ago"
            elif age_seconds < 3600:
                age_str = f"{age_seconds // 60} minutes ago"
            elif age_seconds < 86400:
                age_str = f"{age_seconds // 3600} hours ago"
            else:
                age_str = f"{age_seconds // 86400} days ago"
        else:
            age_str = "Unknown"

        prompt = build_prompt(tx_data)
        explanation = get_ai_explanation(prompt)

        response_data = {
            "hash": tx_hash,
            "explanation": explanation,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "risk_reasons": risk_reasons,
            "gas_fee_eth": f"{gas_fee_eth:.6f}",
            "gas_fee_usd": f"{gas_fee_usd:.2f}" if gas_fee_usd else None,
            "age": age_str,
            "from_addr": tx_data.get("from"),
            "to_addr": tx_data.get("to"),
            "value_eth": f"{int(tx_data.get('value','0x0'),16) / 10**18:.6f}",
            "status": "Success" if tx_data["receipt"].get("status") == "0x1" else "Failed",
            "logs_count": len(tx_data["receipt"].get("logs", [])),
        }
        # Add CORS headers to the response
        response = jsonify(response_data)
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response

    except Exception as e:
        error_response = jsonify({"error": str(e)})
        error_response.headers.add("Access-Control-Allow-Origin", "*")
        return error_response, 500

if __name__ == "__main__":
    print("🚀 Starting Flask server with CORS enabled...")
    app.run(debug=True, host="0.0.0.0", port=5000)        return data["ethereum"]["usd"]
    except:
        return None

# ---------- Helper: fetch block timestamp ----------
def get_block_timestamp(block_number):
    try:
        params = {
            "module": "proxy",
            "action": "eth_getBlockByNumber",
            "blocknumber": hex(block_number),
            "apikey": ETHERSCAN_API_KEY,
            "chainid": 1
        }
        resp = requests.get(ETHERSCAN_BASE, params=params)
        resp.raise_for_status()
        data = resp.json()
        if "result" in data and data["result"]:
            timestamp_hex = data["result"].get("timestamp")
            if timestamp_hex:
                if isinstance(timestamp_hex, int):
                    return timestamp_hex
                if isinstance(timestamp_hex, str):
                    if timestamp_hex.startswith("0x"):
                        return int(timestamp_hex, 16)
                    else:
                        return int(timestamp_hex)
    except:
        pass
    return None

# ---------- Risk scoring ----------
def calculate_risk(tx_data):
    score = 0
    reasons = []
    value_eth = int(tx_data.get("value", "0x0"), 16) / 10**18

    if value_eth > 100:
        score += 30
        reasons.append("Large transfer (>100 ETH)")
    elif value_eth > 10:
        score += 15
        reasons.append("Moderate transfer (>10 ETH)")

    to_addr = tx_data.get("to", "").lower()
    from_addr = tx_data.get("from", "").lower()
    if to_addr == "0x0000000000000000000000000000000000000000":
        score += 20
        reasons.append("Sent to zero address (burn)")
    if from_addr == "0x0000000000000000000000000000000000000000":
        score += 20
        reasons.append("Sent from zero address (mint)")

    input_data = tx_data.get("input", "0x")
    if input_data and input_data != "0x":
        score += 10
        reasons.append("Contract interaction (calldata)")

    logs = tx_data.get("receipt", {}).get("logs", [])
    if len(logs) > 5:
        score += 10
        reasons.append("Many events logged")

    score = min(score, 100)
    if score < 30:
        level = "Low"
    elif score < 60:
        level = "Medium"
    else:
        level = "High"
    return score, level, reasons

# ---------- Etherscan fetcher ----------
def fetch_transaction(tx_hash):
    tx_params = {
        "module": "proxy",
        "action": "eth_getTransactionByHash",
        "txhash": tx_hash,
        "apikey": ETHERSCAN_API_KEY,
        "chainid": 1
    }
    tx_resp = requests.get(ETHERSCAN_BASE, params=tx_params)
    tx_resp.raise_for_status()
    tx_data = tx_resp.json()
    if "error" in tx_data:
        raise Exception(f"Etherscan error: {tx_data['error']}")
    tx = tx_data.get("result")
    if not tx or not isinstance(tx, dict):
        raise Exception("Transaction not found or invalid response.")

    receipt_params = {
        "module": "proxy",
        "action": "eth_getTransactionReceipt",
        "txhash": tx_hash,
        "apikey": ETHERSCAN_API_KEY,
        "chainid": 1
    }
    receipt_resp = requests.get(ETHERSCAN_BASE, params=receipt_params)
    receipt_resp.raise_for_status()
    receipt_data = receipt_resp.json()
    if "error" in receipt_data:
        raise Exception(f"Etherscan receipt error: {receipt_data['error']}")
    receipt = receipt_data.get("result")
    if not receipt or not isinstance(receipt, dict):
        raise Exception("Receipt not found.")
    tx["receipt"] = receipt

    block_num = int(tx.get("blockNumber", "0x0"), 16)
    timestamp = get_block_timestamp(block_num)
    if timestamp is not None:
        tx["blockTimestamp"] = timestamp

    return tx

# ---------- Build prompt ----------
def build_prompt(tx_data):
    tx = tx_data
    receipt = tx.get("receipt", {})
    value_eth = int(tx.get("value", "0x0"), 16) / 10**18
    gas_used = int(receipt.get("gasUsed", "0x0"), 16)
    gas_price = int(tx.get("gasPrice", "0x0"), 16)
    gas_fee_eth = (gas_used * gas_price) / 10**18
    status = "Success" if receipt.get("status") == "0x1" else "Failed"

    prompt = f"""
You are a blockchain transaction explainer. Given the transaction data below, provide a **very short summary** (3‑5 sentences) covering:
- What type of transaction (transfer, swap, contract call, etc.)
- Which assets and how much were moved
- The gas fee and whether it's normal
- Any unusual or notable behavior

Transaction details:
- From: {tx.get('from')}
- To: {tx.get('to')}
- Value: {value_eth:.6f} ETH
- Gas used: {gas_used}
- Gas price: {gas_price} wei
- Gas fee: {gas_fee_eth:.6f} ETH
- Input data (hex): {tx.get('input', '0x')}
- Status: {status}
- Logs: {receipt.get('logs', [])}

Keep it brief – no extra details.
"""
    return prompt

# ---------- AI explanation using gpt-4o-mini (ONLY) ----------
def get_ai_explanation(prompt):
    if not openai_client:
        return "❌ OpenAI API key is missing or invalid. Please set OPENAI_API_KEY in .env"
    try:
        print("🤖 Using OpenAI gpt-4o-mini...")
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a blockchain expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=150
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ OpenAI error: {str(e)}"

# ---------- Flask routes ----------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/explain", methods=["POST"])
def explain():
    tx_hash = request.form.get("tx_hash", "").strip()
    if not tx_hash:
        return jsonify({"error": "Transaction hash is required."}), 400

    try:
        tx_data = fetch_transaction(tx_hash)
        risk_score, risk_level, risk_reasons = calculate_risk(tx_data)

        eth_price = get_eth_usd()
        gas_fee_eth = (int(tx_data["receipt"].get("gasUsed", "0x0"), 16) *
                       int(tx_data.get("gasPrice", "0x0"), 16)) / 10**18
        gas_fee_usd = gas_fee_eth * eth_price if eth_price else None

        timestamp = tx_data.get("blockTimestamp")
        if timestamp is not None:
            if isinstance(timestamp, str):
                try:
                    timestamp = int(timestamp, 16) if timestamp.startswith("0x") else int(timestamp)
                except:
                    timestamp = None

        if timestamp:
            age_seconds = int(time.time() - timestamp)
            if age_seconds < 60:
                age_str = f"{age_seconds} seconds ago"
            elif age_seconds < 3600:
                age_str = f"{age_seconds // 60} minutes ago"
            elif age_seconds < 86400:
                age_str = f"{age_seconds // 3600} hours ago"
            else:
                age_str = f"{age_seconds // 86400} days ago"
        else:
            age_str = "Unknown"

        prompt = build_prompt(tx_data)
        explanation = get_ai_explanation(prompt)

        response_data = {
            "hash": tx_hash,
            "explanation": explanation,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "risk_reasons": risk_reasons,
            "gas_fee_eth": f"{gas_fee_eth:.6f}",
            "gas_fee_usd": f"{gas_fee_usd:.2f}" if gas_fee_usd else None,
            "age": age_str,
            "from_addr": tx_data.get("from"),
            "to_addr": tx_data.get("to"),
            "value_eth": f"{int(tx_data.get('value','0x0'),16) / 10**18:.6f}",
            "status": "Success" if tx_data["receipt"].get("status") == "0x1" else "Failed",
            "logs_count": len(tx_data["receipt"].get("logs", [])),
        }
        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("🚀 Starting Flask server...")
    app.run(debug=True, host="0.0.0.0", port=5000)
