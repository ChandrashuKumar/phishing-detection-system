from flask import Blueprint, request, jsonify
from app.services.url_detection_service import get_detector

url_bp = Blueprint('url', __name__, url_prefix='/api/url')


@url_bp.route('/detect', methods=['POST'])
def detect_phishing():
    """
    Detect phishing for a given URL using dual model architecture

    Request body:
        {
            "url": "https://example.com"
        }

    Response:
        {
            "score": 8.5,
            "model_used": "cross_attention"  // or "url_only"
        }

    Strategy:
        - Runs URL-only model and HTML fetch in parallel
        - If HTML fetch succeeds within 3 seconds, uses cross-attention model (preferred)
        - If HTML fetch fails/times out, uses URL-only model as fallback
        - Returns both score (0-10) and which model was used
    """
    try:
        # Validate request
        data = request.get_json()

        if not data or 'url' not in data:
            return jsonify({'error': 'Missing required field: url'}), 400

        url = data['url']

        if not isinstance(url, str) or len(url.strip()) == 0:
            return jsonify({'error': 'URL must be a non-empty string'}), 400

        # Basic URL validation
        if not (url.startswith('http://') or url.startswith('https://')):
            return jsonify({'error': 'URL must start with http:// or https://'}), 400

        # Get detector and predict
        detector = get_detector()
        result = detector.predict(url)

        return jsonify(result), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
