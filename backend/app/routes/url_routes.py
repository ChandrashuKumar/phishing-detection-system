from flask import Blueprint, request, jsonify
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.services.url_detection_service import get_detector
from app.services.safe_browsing_service import get_safe_browsing_service
from app.services.whois_service import get_whois_service

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

        # Call Google Safe Browsing API and WHOIS API in parallel
        safe_browsing = get_safe_browsing_service()
        whois_service = get_whois_service()

        with ThreadPoolExecutor(max_workers=2) as executor:
            google_future = executor.submit(safe_browsing.check_url, url)
            whois_future = executor.submit(whois_service.get_domain_age, url)

            google_result = google_future.result()
            whois_result = whois_future.result()

        # Extract domain age
        domain_age = whois_result.get('domain_age')

        if google_result['api_success']:
            # Case 1: Google found threats - return immediately
            if google_result['score'] == 10:
                return jsonify({
                    'score': google_result['score'],
                    'threats': google_result['threats'],
                    'domain_age': domain_age
                }), 200

            # Case 2: Google says clean - check ML model for new threats
            detector = get_detector()
            ml_result = detector.predict(url)

            if ml_result['score'] >= ML_THRESHOLD:
                # ML detected potential new threat
                return jsonify({
                    'score': ml_result['score'],
                    'threats': [],
                    'domain_age': domain_age
                }), 200
            else:
                # Trust Google's clean verdict
                return jsonify({
                    'score': 0,
                    'threats': [],
                    'domain_age': domain_age
                }), 200

        # Case 3: Google API failed - fallback to ML model
        detector = get_detector()
        ml_result = detector.predict(url)

        return jsonify({
            'score': ml_result['score'],
            'threats': [],
            'domain_age': domain_age
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
