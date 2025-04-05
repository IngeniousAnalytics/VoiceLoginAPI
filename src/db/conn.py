import os
from dotenv import load_dotenv
from typing import Dict

class DatabaseConfig:
    """Secure database configuration manager"""
    
    def __init__(self):
        load_dotenv()
        self._validate_env_vars()
        
    def _validate_env_vars(self) -> None:
        """Validate required environment variables"""
        required_vars = ['DB_NAME', 'DB_USER', 'DB_PASSWORD', 'DB_HOST', 'DB_PORT']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            raise EnvironmentError(
                f"Missing required database variables: {', '.join(missing_vars)}"
            )

    @property
    def config(self) -> Dict[str, str]:
        """Get validated database configuration"""
        return {
            'dbname': os.getenv('DB_NAME'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD'),
            'host': os.getenv('DB_HOST'),
            'port': os.getenv('DB_PORT')
        }

    @property
    def url(self) -> str:
        """Construct secured database URL"""
        config = self.config
        return (
            f"postgresql+asyncpg://"
            f"{config['user']}:{config['password']}"
            f"@{config['host']}:{config['port']}"
            f"/{config['dbname']}"
            "?ssl=true"  # Force SSL for production
        )

# Initialize configuration
db_config = DatabaseConfig()
DATABASE_URL = db_config.url
DB_CONFIG = db_config.config