import requests
from typing import Dict, Optional
from urllib.parse import urlparse
from datetime import datetime


class WhoisService:
    """Service for checking domain age using APILayer WHOIS API"""

    def __init__(self, api_key: str):
        """
        Initialize WHOIS service

        Args:
            api_key: APILayer API key
        """
        self.api_key = api_key
        self.api_url = "https://api.apilayer.com/whois/query"

    def extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path
        # Remove www. prefix if present
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain

    def calculate_domain_age(self, creation_date_str: str) -> Optional[int]:
        """Calculate domain age in days from creation date string"""
        try:
            # Parse creation date (format: "1997-09-15 04:00:00+00:00")
            creation_date = datetime.strptime(creation_date_str.split('+')[0].strip(), "%Y-%m-%d %H:%M:%S")
            now = datetime.now()
            age_days = (now - creation_date).days
            return age_days
        except Exception:
            return None

    def get_domain_age(self, url: str) -> Dict:
        """
        Get domain age for a URL

        Args:
            url: URL to check

        Returns:
            dict: {
                'domain_age': int (days) or None if failed,
                'api_success': bool
            }
        """
        try:
            domain = self.extract_domain(url)

            if not domain:
                return {
                    'domain_age': None,
                    'api_success': False,
                    'error': 'Could not extract domain from URL'
                }

            # Make API request
            response = requests.get(
                f"{self.api_url}?domain={domain}",
                headers={'apikey': self.api_key},
                timeout=5
            )

            if response.status_code == 200:
                data = response.json()

                # Check if result exists
                if 'result' not in data:
                    return {
                        'domain_age': None,
                        'api_success': False,
                        'error': 'No result in response'
                    }

                result = data['result']
                creation_date = result.get('creation_date')

                if creation_date:
                    domain_age = self.calculate_domain_age(creation_date)
                    if domain_age is not None:
                        return {
                            'domain_age': domain_age,
                            'api_success': True
                        }
                    else:
                        return {
                            'domain_age': None,
                            'api_success': False,
                            'error': 'Could not parse creation date'
                        }
                else:
                    return {
                        'domain_age': None,
                        'api_success': False,
                        'error': 'creation_date not found in response'
                    }
            else:
                return {
                    'domain_age': None,
                    'api_success': False,
                    'error': f"API returned status {response.status_code}"
                }

        except requests.Timeout:
            return {
                'domain_age': None,
                'api_success': False,
                'error': "API request timeout"
            }
        except Exception as e:
            return {
                'domain_age': None,
                'api_success': False,
                'error': str(e)
            }


# Global instance
_whois_service = None


def get_whois_service():
    """Get or create the global WHOIS service instance"""
    global _whois_service

    if _whois_service is None:
        from config import Config
        api_key = Config.APILAYER_API_KEY

        if not api_key:
            raise ValueError("APILAYER_API_KEY not found in config")

        _whois_service = WhoisService(api_key)
        print("WHOIS service initialized")

    return _whois_service
