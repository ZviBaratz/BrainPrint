import logging
import sys

LOG_FILE_NAME: str = "brainprint.log"

file_handler = logging.FileHandler(filename=LOG_FILE_NAME)
stdout_handler = logging.StreamHandler(sys.stdout)
handlers = [file_handler, stdout_handler]

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    handlers=handlers,
)

# Disable annoying debug messages from other module.
logging.getLogger("blib2to3").setLevel(logging.WARNING)
logging.getLogger("parso").setLevel(logging.WARNING)
