from flask import Blueprint, request, jsonify
from app.services.url_detection_service import get_detector
from app.services.safe_browsing_service import get_safe_browsing_service

url_bp = Blueprint('url', __name__, url_prefix='/api/url')

ML_THRESHOLD = 7.0  # Threshold for trusting ML model on new threats


@url_bp.route('/detect', methods=['POST'])
def detect_phishing():
    """URL phishing detection with smart hybrid approach"""
    try:
        data = request.get_json()

        if not data or 'url' not in data:
            return jsonify({'error': 'Missing required field: url'}), 400

        url = data['url']

        if not isinstance(url, str) or len(url.strip()) == 0:
            return jsonify({'error': 'URL must be a non-empty string'}), 400

        if not (url.startswith('http://') or url.startswith('https://')):
            return jsonify({'error': 'URL must start with http:// or https://'}), 400

        # Try Google Safe Browsing API
        safe_browsing = get_safe_browsing_service()
        google_result = safe_browsing.check_url(url)

        if google_result['api_success']:
            # Case 1: Google found threats - return immediately
            if google_result['score'] == 10:
                return jsonify({
                    'score': google_result['score'],
                    'threats': google_result['threats']
                }), 200

            # Case 2: Google says clean - check ML model for new threats
            detector = get_detector()
            ml_result = detector.predict(url)

            if ml_result['score'] >= ML_THRESHOLD:
                # ML detected potential new threat
                return jsonify({
                    'score': ml_result['score'],
                    'threats': []
                }), 200
            else:
                # Trust Google's clean verdict
                return jsonify({
                    'score': 0,
                    'threats': []
                }), 200

        # Case 3: Google API failed - fallback to ML model
        detector = get_detector()
        ml_result = detector.predict(url)

        return jsonify({
            'score': ml_result['score'],
            'threats': []
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
