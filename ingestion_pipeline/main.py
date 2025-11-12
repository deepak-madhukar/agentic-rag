import asyncio
import logging
from ingestion_pipeline import main as run_ingestion

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    asyncio.run(run_ingestion())
