import requests
from typing import Dict, List


class SafeBrowsingService:
    """Service for checking URLs against Google Safe Browsing API v5"""

    def __init__(self, api_key: str):
        """
        Initialize Safe Browsing service

        Args:
            api_key: Google Safe Browsing API key
        """
        self.api_key = api_key
        self.api_url = "https://safebrowsing.googleapis.com/v5alpha1/urls:search"

    def check_url(self, url: str) -> Dict:
        """
        Check if URL is flagged by Google Safe Browsing

        Args:
            url: URL to check (max 50 URLs, but we check 1 at a time)

        Returns:
            dict: {
                'score': int (0 or 10) or None if API failed,
                'threats': list of threat types found,
                'api_success': bool,
                'cache_duration': str (optional)
            }
        """
        try:
            # Prepare query parameters
            params = {
                'urls': url,
                'key': self.api_key
            }

            # Make API request
            response = requests.get(
                self.api_url,
                params=params,
                timeout=5
            )

            # Check response
            if response.status_code == 200:
                data = response.json()

                # Check if threats were found
                if "threats" in data and len(data["threats"]) > 0:
                    # URL is flagged - extract threat types
                    threat_types = []
                    for threat in data["threats"]:
                        if "threatTypes" in threat:
                            threat_types.extend(threat["threatTypes"])

                    return {
                        'score': 10,  # Flagged as malicious
                        'threats': list(set(threat_types)),  # Remove duplicates
                        'api_success': True,
                        'cache_duration': data.get('cacheDuration', '3600s')
                    }
                else:
                    # No threats found - URL is clean
                    return {
                        'score': 0,  # Clean
                        'threats': [],
                        'api_success': True,
                        'cache_duration': data.get('cacheDuration', '3600s')
                    }
            else:
                # API error
                return {
                    'score': None,
                    'threats': [],
                    'api_success': False,
                    'error': f"API returned status {response.status_code}: {response.text}"
                }

        except requests.Timeout:
            return {
                'score': None,
                'threats': [],
                'api_success': False,
                'error': "API request timeout"
            }
        except Exception as e:
            return {
                'score': None,
                'threats': [],
                'api_success': False,
                'error': str(e)
            }


# Global instance
_safe_browsing_service = None


def get_safe_browsing_service():
    """Get or create the global Safe Browsing service instance"""
    global _safe_browsing_service

    if _safe_browsing_service is None:
        from config import Config
        api_key = Config.GOOGLE_SAFE_BROWSING_API_KEY

        if not api_key:
            raise ValueError("GOOGLE_SAFE_BROWSING_API_KEY not found in config")

        _safe_browsing_service = SafeBrowsingService(api_key)
        print("Safe Browsing service initialized")

    return _safe_browsing_service
