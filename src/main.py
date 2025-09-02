from fastapi import FastAPI

app = FastAPI(title="Plagiarism API")


@app.get("/")
async def root():
    return {"status": "Ok"}
