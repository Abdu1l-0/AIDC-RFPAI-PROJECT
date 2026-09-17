from fastapi import FastAPI

app = FastAPI(title="RFP Benchmark Middleware")

@app.get("/health")
def health():
        return {"status": "ok"}