import os
from flask import Flask, request, jsonify
from openai import OpenAI
from PIL import Image
import base64
import requests
from io import BytesIO
from flask_cors import CORS
import logging

app = Flask(__name__)
CORS(app)

# Configure your OpenAI API key
os.environ["OPENAI_API_KEY"] = ""
STABILITY_API_KEY = ""

# Initialize OpenAI client
client = OpenAI()

# Configure logging
logging.basicConfig(level=logging.DEBUG, filename='server.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

@app.route('/describe_sketch', methods=['POST'])
def describe_sketch():
    try:
        logging.debug("Received request to describe sketch.")

        # Debug: Check if request contains an image file
        if 'image' not in request.files:
            logging.error("No image file provided in the request.")
            return jsonify({"error": "No image file provided."}), 400

        # Debug: Try processing the image
        try:
            image_file = request.files['image']
            image = Image.open(image_file)
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
        except Exception as e:
            logging.error(f"Failed to process image: {str(e)}")
            return jsonify({"error": f"Failed to process image: {str(e)}"}), 400

        # Debug: Query GPT-4 Vision with image
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "I want a 1 sentence description of this sketch, be laconic and emphasize brevity. Do not include any response details like \"Description:\".",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
            )
            description = str(response.choices[0].message.content)
            logging.info(f"Description: {description}")
        except Exception as e:
            logging.error(f"OpenAI API Error: {str(e)}")
            return jsonify({"error": f"OpenAI API Error: {str(e)}"}), 500

        # Debug: Use the description to label and send to Stability AI
        try:
            response = requests.post(
                "https://api.stability.ai/v2beta/stable-image/control/sketch",
                headers={
                    "Authorization": f"Bearer {STABILITY_API_KEY}",
                    "Accept": "image/*"  # Correctly set the Accept header
                },
                files={
                    "image": ("sketch.png", BytesIO(base64.b64decode(base64_image)), "image/png")
                },
                data={
                    "prompt": description
                }
            )

            if response.status_code == 200:
                transformed_image_b64 = base64.b64encode(response.content).decode("utf-8")
            else:
                logging.error(f"Unexpected Stability AI response: {response.json()}")
                return jsonify({"error": f"Unexpected Stability AI response: {response.json()}"}), 500
        except Exception as e:
            logging.error(f"Stability AI Error: {str(e)}")
            return jsonify({"error": f"Stability AI Error: {str(e)}"}), 500

        logging.info("Request processed successfully.")
        return jsonify({
            "description": description,
            "transformed_image": transformed_image_b64
        })

    except Exception as e:
        logging.critical(f"Unhandled error: {str(e)}")
        return jsonify({"error": f"Unhandled error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
