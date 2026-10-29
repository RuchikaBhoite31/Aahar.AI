import os, tempfile
os.environ["AAHAR_DB"] = os.path.join(tempfile.mkdtemp(), "test.db")   # isolate DB before app import
os.environ["AAHAR_LLM_PROVIDER"] = "none"                              # deterministic, offline
