import traceback
from src.retrieve import run_retrieval_pipeline

try:
    run_retrieval_pipeline('test')
except Exception as e:
    print("Caught exception:")
    traceback.print_exc()
