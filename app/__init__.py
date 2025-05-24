from flask import Flask


def create_app():
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        UPLOAD_FOLDER='uploads',
        OUTPUT_FOLDER='static/results'
    )

    # Importer les routes
    from .routes import routes_bp
    app.register_blueprint(routes_bp)

    return app