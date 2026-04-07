from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def root():
    return {"message": "APP IS WORKING"}


@app.get("/check")
def check():
    return {"status": "OK"}


@app.get("/hello")
def hello():
    return {"msg": "hello world"}
