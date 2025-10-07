import os, json, uuid, base64, datetime
from flask import Flask, request, send_from_directory, jsonify
from flask_cors import CORS
import requests
from PIL import Image
import pytesseract
import speech_recognition as sr
import pyttsx3
import torch
import torchvision.transforms as transforms
from torchvision import models

app = Flask(__name__, static_folder="frontend")
CORS(app)

DATA_DIR = "data"
CHATS_DIR = os.path.join(DATA_DIR, "chats")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
LOGS_DIR = os.path.join(DATA_DIR, "logs")
TRANSCRIPTS_DIR = os.path.join(DATA_DIR, "transcripts")
SETTINGS_PATH = os.path.join(DATA_DIR, "settings.json")
CONFIG_PATH = "config.json"

os.makedirs(CHATS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)

def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)

def save_chat(chat_id, chat_data):
    path = os.path.join(CHATS_DIR, f"chat_{chat_id}.json")
    with open(path, "w") as f:
        json.dump(chat_data, f, indent=2)

def save_log(chat_id, text):
    path = os.path.join(LOGS_DIR, f"chat_{chat_id}.log")
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + "\n")

def save_transcript(chat_id, text):
    path = os.path.join(TRANSCRIPTS_DIR, f"chat_{chat_id}.txt")
    with open(path, "a") as f:
        f.write(text + "\n")

def save_image(img_b64):
    img_bytes = base64.b64decode(img_b64)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"img_{ts}.png"
    path = os.path.join(IMAGES_DIR, fname)
    with open(path, "wb") as f:
        f.write(img_bytes)
    return fname

def ocr_image(img_path):
    try:
        import pytesseract
        img = Image.open(img_path)
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        # If Tesseract is not installed, skip OCR and return a message
        return "[OCR unavailable: Tesseract not installed]"

def image_recognition(img_path):
    # Load pretrained ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.eval()
    # ImageNet labels
    labels_path = os.path.join(os.path.dirname(__file__), "imagenet_classes.txt")
    if not os.path.exists(labels_path):
        # Download labels file if missing
        import urllib.request
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
            labels_path
        )
    with open(labels_path) as f:
        labels = [line.strip() for line in f.readlines()]
    # Preprocess image
    input_image = Image.open(img_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        output = model(input_batch)
    _, predicted = torch.max(output, 1)
    label = labels[predicted.item()]
    return label

@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory("frontend", path)

@app.route("/api/chats", methods=["GET"])
def list_chats():
    chats = []
    for fname in os.listdir(CHATS_DIR):
        if fname.endswith(".json"):
            with open(os.path.join(CHATS_DIR, fname)) as f:
                data = json.load(f)
                chats.append({"id": data.get("id"), "name": data.get("name")})
    return jsonify(chats)

@app.route("/api/chat/<chat_id>", methods=["GET"])
def get_chat(chat_id):
    path = os.path.join(CHATS_DIR, f"chat_{chat_id}.json")
    if not os.path.exists(path): return jsonify({})
    with open(path) as f:
        return jsonify(json.load(f))

@app.route("/api/chat", methods=["POST"])
def new_chat():
    data = request.json
    chat_id = str(uuid.uuid4())[:8]
    name = data.get("name", f"United Chat #{len(os.listdir(CHATS_DIR))+1}")
    chat_data = {"id": chat_id, "name": name, "messages": []}
    save_chat(chat_id, chat_data)
    return jsonify(chat_data)

@app.route("/api/chat/<chat_id>", methods=["DELETE"])
def delete_chat(chat_id):
    path = os.path.join(CHATS_DIR, f"chat_{chat_id}.json")
    if os.path.exists(path): os.remove(path)
    log_path = os.path.join(LOGS_DIR, f"chat_{chat_id}.log")
    if os.path.exists(log_path): os.remove(log_path)
    return jsonify({"ok": True})

@app.route("/api/ask", methods=["POST"])
def ask():
    payload = request.json
    chat_id = payload.get("chat_id")
    message = payload.get("message", "")
    image_b64 = payload.get("image")
    image_text = ""
    image_label = ""
    image_error = ""
    if image_b64:
        fname = save_image(image_b64)
        img_path = os.path.join(IMAGES_DIR, fname)
        # OCR (skip if not available)
        image_text = ocr_image(img_path)
        # Recognition
        try:
            image_label = image_recognition(img_path)
        except Exception as e:
            image_label = ""
            image_error = f"Image recognition failed: {str(e)}"
        # Compose message for LLM
        if image_text.strip():
            message += "\n[Image text: " + image_text + "]"
        if image_label:
            message += f"\n[Image label: {image_label}]"
        elif image_error:
            message += f"\n[Image label error: {image_error}]"
    config = load_config()
    api_url = f"{config['base_url']}/chat/completions"
    headers = {"Authorization": f"Bearer {config['api_key']}"}
    body = {
        "model": config["model_name"],
        "messages": [{"role": "user", "content": message}]
    }
    try:
        resp = requests.post(api_url, headers=headers, json=body)
        result = resp.json()
        reply = result.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception as e:
        reply = f"Error: Could not get response from server. {str(e)}"
    # Save chat
    path = os.path.join(CHATS_DIR, f"chat_{chat_id}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            chat_data = json.load(f)
        chat_data["messages"].append({"role": "user", "content": message})
        # Always include image label and OCR in assistant message if present
        assistant_msg = reply
        if image_label:
            assistant_msg += f"\n\n[Image label: {image_label}]"
        if image_text:
            assistant_msg += f"\n[Image text: {image_text}]"
        if image_error:
            assistant_msg += f"\n{image_error}"
        chat_data["messages"].append({"role": "assistant", "content": assistant_msg})
        with open(path, "w", encoding="utf-8") as f:
            json.dump(chat_data, f, indent=2, ensure_ascii=False)
    save_log(chat_id, f"User: {message}\nAssistant: {reply}")
    # Always return image label and OCR in API response
    return jsonify({
        "reply": reply,
        "image_text": image_text,
        "image_label": image_label,
        "image_error": image_error
    })

@app.route("/api/ocr", methods=["POST"])
def ocr():
    data = request.json
    img_b64 = data.get("image")
    fname = save_image(img_b64)
    img_path = os.path.join(IMAGES_DIR, fname)
    text = ocr_image(img_path)
    return jsonify({"text": text})

@app.route("/api/stt", methods=["POST"])
def stt():
    # Accepts audio file, returns transcript
    audio = request.files["audio"]
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
        except Exception:
            text = ""
    return jsonify({"text": text})

@app.route("/api/tts", methods=["POST"])
def tts():
    data = request.json
    text = data.get("text", "")
    engine = pyttsx3.init()
    engine.save_to_file(text, "tts_output.mp3")
    engine.runAndWait()
    return send_from_directory(".", "tts_output.mp3")

@app.route("/api/settings", methods=["GET", "POST"])
def settings():
    if request.method == "GET":
        if not os.path.exists(SETTINGS_PATH): return jsonify({})
        with open(SETTINGS_PATH) as f:
            return jsonify(json.load(f))
    else:
        data = request.json
        with open(SETTINGS_PATH, "w") as f:
            json.dump(data, f, indent=2)
        return jsonify({"ok": True})

if __name__ == "__main__":
    app.run(port=5000)
