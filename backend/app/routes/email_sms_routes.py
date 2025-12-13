from flask import Blueprint, request, jsonify
from app.services.email_sms_service import get_detector

email_sms_bp = Blueprint('email_sms', __name__, url_prefix='/api/email-sms')


@email_sms_bp.route('/detect', methods=['POST'])
def detect_phishing():
    """
    Detect phishing in email/SMS text

    Request JSON:
        {
            "text": "Email or SMS content here..."
        }

    Response JSON:
        {
            "score": 7.5
        }
    """
    try:
        # Get request data
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({
                'error': 'Missing required field: text'
            }), 400

        text = data['text']

        # Validate text
        if not isinstance(text, str) or len(text.strip()) == 0:
            return jsonify({
                'error': 'Text must be a non-empty string'
            }), 400

        # Get prediction
        detector = get_detector()
        score = detector.predict(text)

        return jsonify({'score': score}), 200

    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500
