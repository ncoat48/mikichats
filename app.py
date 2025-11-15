import os
import random
import string
import re
import json
import io
import base64
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
from dotenv import load_dotenv
from datetime import datetime, timezone

import google.generativeai as genai

# --- Firebase ---
import firebase_admin
from firebase_admin import credentials, firestore

# --- Stability AI ---
import stability_sdk.client as StabilityClient
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation

# --- Cloudinary ---
import cloudinary
import cloudinary.uploader
import cloudinary.api

# --- VERCEL FIX 1: Load Environment Variables ---
load_dotenv() 

# --- VERCEL FIX 2: Handle serviceAccountKey.json ---
service_account_json_string = os.environ.get('FIREBASE_SERVICE_ACCOUNT_JSON')

try:
    if service_account_json_string:
        service_account_info = json.loads(service_account_json_string)
        cred = credentials.Certificate(service_account_info)
    else:
        cred = credentials.Certificate('serviceAccountKey.json')

    firebase_admin.initialize_app(cred)
except FileNotFoundError:
    print("="*50)
    print("ERROR: serviceAccountKey.json not found (for local dev).")
    print("="*50)
except ValueError as e:
    print(f"Firebase already initialized? {e}")
    pass 
except json.JSONDecodeError:
    print("="*50)
    print("ERROR: FIREBASE_SERVICE_ACCOUNT_JSON is not valid JSON.")
    print("="*50)

db = firestore.client()
# --- END OF VERCEL FIXES ---


# --- VERCEL FIX 3: Tell Flask where the 'templates' folder is ---
base_dir = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__,
            template_folder=os.path.join(base_dir, 'templates')
           )
# --- END OF VERCEL FIX ---

app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'a_very_strong_default_secret_key_12345')

# --- Configure Gemini ---
api_key = os.environ.get('GEMINI_API_KEY')
if not api_key:
    raise ValueError("GEMINI_API_KEY not found. Set it in Vercel Environment Variables.")
genai.configure(api_key=api_key) 
model = genai.GenerativeModel('gemini-2.5-flash')

# --- Configure Stability AI ---
stability_api_key = os.environ.get("STABILITY_API_KEY")
if not stability_api_key:
    raise ValueError("STABILITY_API_KEY not found. Set it in Vercel Environment Variables.")
stability_api = StabilityClient.StabilityInference(
    key=stability_api_key,
    verbose=True,
    engine="stable-diffusion-xl-1024-v1-0",
)

# --- Configure Cloudinary ---
cloudinary.config( 
    cloud_name = os.environ.get("CLOUDINARY_CLOUD_NAME"), 
    api_key = os.environ.get("CLOUDINARY_API_KEY"), 
    api_secret = os.environ.get("CLOUDINARY_API_SECRET") 
)
if not os.environ.get("CLOUDINARY_CLOUD_NAME"):
    raise ValueError("CLOUDINARY_CLOUD_NAME not found. Set it in Vercel Environment Variables.")


# --- Helper Functions (Unchanged) ---
def generate_room_code(length=4):
    while True:
        code = ''.join(random.choices(string.digits, k=length))
        room_ref = db.collection('rooms').document(code)
        if not room_ref.get().exists:
            return code

def get_user_id():
    if 'user_id' not in session:
        session['user_id'] = f"user_{random.randint(1000, 9999)}"
    return session['user_id']

def get_nickname():
    return session.get('nickname', 'Anonymous')

# --- build_gemini_prompt (Unchanged) ---
def build_gemini_prompt(room_data, user_nickname, message_text):
    prompt_lines = [
        f"You are {room_data['bot_name']}. Your personality is: {room_data['bot_personality']}.",
        f"Your appearance is: {room_data.get('bot_appearance', 'not specified')}",
        "You are in a role-playing game where multiple users are trying to win your affection.",
    ]
    prompt_lines.append("\nCurrent affection levels:")
    if not room_data.get('users'):
        prompt_lines.append("No one is in the room yet.")
    else:
        for user_id, data in room_data['users'].items():
            prompt_lines.append(f"- {data['nickname']}: {data['score']}%")

    prompt_lines.append(f"\nThe current scenario: {room_data['start_scenario']}")
    prompt_lines.append("\nHere is the recent chat history (max 10):")
    
    sorted_messages = sorted(room_data.get('messages', []), key=lambda m: m.get('timestamp'))
    
    for msg in sorted_messages[-10:]:
        sender = msg['user']
        if sender == room_data['bot_name']:
            sender_display = "You"
        elif sender == "System":
            sender_display = "System"
        else:
            sender_display = next((data['nickname'] for uid, data in room_data.get('users', {}).items() if uid == sender), sender)
        
        prompt_lines.append(f"{sender_display}: {msg['text']}")

    prompt_lines.append(f"\n--- NEW MESSAGE ---")
    prompt_lines.append(f"{user_nickname}: {message_text}")
    prompt_lines.append("\n--- YOUR TASK ---")
    prompt_lines.append(
        "Based on this new message, you must do two things:"
        f"\n1.  **Respond** in character as {room_data['bot_name']}."
        "\n2.  **Evaluate** the user's message. How much did it change your affection for them?"
        f" The difficulty is {room_data['difficulty']}/10."
        "\nYou MUST reply in this exact JSON format (no markdown):"
        "\n{"
        "\n  \"response\": \"Your in-character reply here.\","
        "\n  \"affection_change\": <number from -20 to 20>"
        "\n}"
    )
    return "\n".join(prompt_lines)

# --- parse_gemini_response (Unchanged) ---
def parse_gemini_response(text):
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if not match:
            print(f"Parsing Error: No JSON block found in response. Response was: {text}")
            return {"response": text, "affection_change": 0}
        json_text = match.group(0)
        data = json.loads(json_text)
        return {
            "response": data.get("response", "I'm not sure what to say."),
            "affection_change": int(data.get("affection_change", 0))
        }
    except Exception as e:
        print(f"Error parsing Gemini response: {e} - Response was: {text}")
        return {"response": text, "affection_change": 0}

# --- index route (Unchanged) ---
@app.route("/")
def index():
    return render_template("index.html")

# --- /generate-bot-image (Unchanged) ---
@app.route("/generate-bot-image", methods=["POST"])
def generate_bot_image():
    try:
        data = request.get_json()
        gender = data.get('gender', 'person')
        age = data.get('age', '20')
        appearance = data.get('appearance', 'average')

        prompt = f"A beautiful portrait of a {age} year old {gender}, {appearance}. digital art, anime style, detailed face, cinematic lighting, high quality"
        negative_prompt = "blurry, deformed, ugly, bad anatomy, mutated, extra limbs, disfigured"
        
        print(f"Stability Prompt: {prompt}")

        answers = stability_api.generate(
            prompt=[
                generation.Prompt(text=prompt, parameters=generation.PromptParameters(weight=1.0)),
                generation.Prompt(text=negative_prompt, parameters=generation.PromptParameters(weight=-1.0))
            ],
            style_preset="anime",
            steps=30,
            cfg_scale=7.0,
            width=512,
            height=512,
            samples=1
        )

        for resp in answers:
            for artifact in resp.artifacts:
                if artifact.finish_reason == generation.FILTER:
                    return jsonify({'success': False, 'error': 'Image generation was filtered for safety. Try a different prompt.'}), 400
                
                if artifact.type == generation.ARTIFACT_IMAGE:
                    img_data = artifact.binary
                    
                    print("Uploading to Cloudinary...")
                    upload_result = cloudinary.uploader.upload(
                        img_data,
                        folder="bot_avatars",
                        public_id=f"bot_{generate_room_code(10)}"
                    )
                    
                    secure_url = upload_result.get('secure_url')
                    if not secure_url:
                        return jsonify({'success': False, 'error': 'Failed to upload image to Cloudinary.'}), 500
                    
                    print(f"Image uploaded: {secure_url}")
                    return jsonify({'success': True, 'image_url': secure_url})

        return jsonify({'success': False, 'error': 'No image was generated.'}), 500

    except Exception as e:
        print(f"Error generating image: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# --- /upload-bot-image (Unchanged) ---
@app.route("/upload-bot-image", methods=["POST"])
def upload_bot_image():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'}), 400
    
    if file:
        try:
            print("Uploading custom image to Cloudinary...")
            upload_result = cloudinary.uploader.upload(
                file,
                folder="bot_avatars",
                public_id=f"bot_{generate_room_code(10)}"
            )
            secure_url = upload_result.get('secure_url')
            if not secure_url:
                return jsonify({'success': False, 'error': 'Failed to upload to Cloudinary.'}), 500
            
            print(f"Image uploaded: {secure_url}")
            return jsonify({'success': True, 'image_url': secure_url})
        
        except Exception as e:
            print(f"Error uploading image: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500

    return jsonify({'success': False, 'error': 'Unknown file error'}), 500

# --- create_room (THIS IS THE FIX) ---
@app.route("/create", methods=["GET", "POST"])
def create_room():
    if request.method == "POST":
        room_code = generate_room_code()
        user_id = get_user_id()
        
        nickname = request.form.get('nickname', 'Host')
        if not nickname.strip():
            nickname = 'Host'
        session['nickname'] = nickname
        
        bot_name = request.form.get('bot_name', 'Bot')
        start_scenario = request.form.get('start_scenario', 'You meet at a park.')
        
        bot_image_url = request.form.get('bot_image_url') or "https://placehold.co/100x100/4a5568/FFFFFF?text=Bot"
        bot_appearance = request.form.get('appearance', 'not specified')

        py_timestamp = datetime.utcnow().replace(tzinfo=timezone.utc) 

        # --- THIS IS THE FIX ---
        # The typo `new_room_.data` is now `new_room_data`
        new_room_data = {
            'bot_name': bot_name,
            'bot_personality': request.form.get('bot_personality', 'friendly'),
            'start_scenario': start_scenario,
            'difficulty': int(request.form.get('difficulty', 5)),
            'game_over': False,
            'users': {
                user_id: {'nickname': nickname, 'score': 0}
            },
            'messages': [
                 {
                    'user': 'System', 
                    'text': f"Game started! {nickname} created the room.",
                    'timestamp': py_timestamp 
                },
                {
                    'user': bot_name, 
                    'text': start_scenario,
                    'timestamp': py_timestamp 
                }
            ],
            'bot_image_url': bot_image_url,
            'bot_appearance': bot_appearance
        }
        # --- END OF FIX ---
        
        db.collection('rooms').document(room_code).set(new_room_data)
        
        return redirect(url_for('chat_room', room_code=room_code))
    
    return render_template("create.html")

# --- join_room (Unchanged) ---
@app.route("/join", methods=["POST"])
def join_room():
    room_code = request.form.get('room_code')
    nickname = request.form.get('nickname', 'Guest')
    if not nickname.strip():
        nickname = f"Guest_{random.randint(100, 999)}"
    
    room_ref = db.collection('rooms').document(room_code)
    room_doc = room_ref.get()

    if not room_doc.exists:
        flash("Error: Room code not found.")
        return redirect(url_for('index'))
    
    room_data = room_doc.to_dict()
        
    if room_data.get('game_over', False):
        flash("Error: This game has already ended.")
        return redirect(url_for('index'))
        
    user_id = get_user_id()
    session['nickname'] = nickname
    
    if user_id not in room_data['users']:
        
        py_timestamp = datetime.utcnow().replace(tzinfo=timezone.utc)
        
        room_ref.update({
            f'users.{user_id}': {'nickname': nickname, 'score': 0},
            'messages': firestore.ArrayUnion([
                {
                    'user': 'System',
                    'text': f"{nickname} has joined the game!",
                    'timestamp': py_timestamp 
                }
            ])
        })
        
    return redirect(url_for('chat_room', room_code=room_code))

# --- chat_room (Unchanged) ---
@app.route("/room/<room_code>", methods=["GET", "POST"])
def chat_room(room_code):
    room_ref = db.collection('rooms').document(room_code)
    room_doc = room_ref.get()

    if not room_doc.exists:
        flash("Error: Room not found.")
        return redirect(url_for('index'))
        
    user_id = get_user_id()
    nickname = get_nickname()
    room_data = room_doc.to_dict()
    
    if user_id not in room_data.get('users', {}):
        flash("Error: You are not part of this room. Please join first.")
        return redirect(url_for('index'))

    # Handle a new message submission
    if request.method == "POST":
        if room_data.get('game_over', False):
            return jsonify({'success': False, 'error': 'Game is over'}), 400

        message_text = request.form.get('message')
        if not message_text:
            return jsonify({'success': False, 'error': 'Empty message'}), 400
            
        try:
            py_timestamp_start = datetime.utcnow().replace(tzinfo=timezone.utc)

            prompt = build_gemini_prompt(room_data, nickname, message_text)
            response = model.generate_content(prompt)
            parsed_data = parse_gemini_response(response.text)
            
            bot_response = parsed_data['response']
            affection_change = parsed_data['affection_change']

            py_timestamp_end = datetime.utcnow().replace(tzinfo=timezone.utc)

            user_data = room_data['users'][user_id]
            new_score = user_data['score'] + affection_change
            new_score = max(0, min(100, new_score))
            
            score_msg = f"{nickname}'s affection didn't change. (Still {new_score}%)"
            if affection_change > 0:
                score_msg = f"{nickname}'s affection went up by {affection_change}%! (Now {new_score}%)"
            elif affection_change < 0:
                score_msg = f"{nickname}'s affection went down by {abs(affection_change)}%! (Now {new_score}%)"

            messages_to_add = [
                {'user': user_id, 'text': message_text, 'timestamp': py_timestamp_start},
                {'user': room_data['bot_name'], 'text': bot_response, 'timestamp': py_timestamp_end},
                {'user': 'System', 'text': score_msg, 'timestamp': py_timestamp_end}
            ]

            update_data = {
                f'users.{user_id}.score': new_score
            }

            if new_score >= 100:
                update_data['game_over'] = True
                messages_to_add.append({
                    'user': 'System',
                    'text': f"GAME OVER! {nickname} has won {room_data['bot_name']}'s affection!",
                    'timestamp': py_timestamp_end
                })
            
            update_data['messages'] = firestore.ArrayUnion(messages_to_add)
            room_ref.update(update_data)
            
            return jsonify({'success': True})

        except Exception as e:
            print(f"Error during Gemini call: {e}")
            py_timestamp_error = datetime.utcnow().replace(tzinfo=timezone.utc)
            room_ref.update({
                'messages': firestore.ArrayUnion([
                    {'user': user_id, 'text': message_text, 'timestamp': py_timestamp_error},
                    {'user': 'System', 'text': f"Sorry, {nickname}, I'm having trouble thinking. (Error: {e})", 'timestamp': py_timestamp_error}
                ])
            })
            return jsonify({'success': False, 'error': str(e)}), 500

    return render_template("room.html", room=room_data, room_code=room_code, user_id=user_id)
