from flask import Flask
from flask_cors import CORS
from config import Config


def create_app(config_class=Config):
    """Flask app factory"""
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Enable CORS
    CORS(app)

    # Register blueprints
    from app.routes.email_sms_routes import email_sms_bp
    app.register_blueprint(email_sms_bp)

    @app.route('/health')
    def health_check():
        return {'status': 'healthy'}, 200

    return app
