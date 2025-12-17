import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Base configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'

    # Model paths
    MODEL_DIR = os.environ.get('MODEL_DIR') or os.path.join(os.path.dirname(__file__), 'models')

    # Google Safe Browsing API
    GOOGLE_SAFE_BROWSING_API_KEY = os.environ.get('GOOGLE_SAFE_BROWSING_API_KEY')

    # APILayer WHOIS API
    APILAYER_API_KEY = os.environ.get('APILAYER_API_KEY')

    # API settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max request size

    # CORS settings
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*').split(',')


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    FLASK_ENV = 'development'


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    FLASK_ENV = 'production'


config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
