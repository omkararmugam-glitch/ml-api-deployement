from fastapi import FastAPI

app = FastAPI(
    title="Iris API",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)


@app.get("/")
def root():
    return {"message": "FINAL WORKING VERSION"}


@app.get("/check")
def check():
    return {"status": "OK"}


@app.get("/hello")
def hello():
    return {"msg": "hello world"}
