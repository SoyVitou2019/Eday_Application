import cv2
import uvicorn
import jsonpickle
import numpy as np
from fastapi import FastAPI, Request, Response
from model import XG_predict

# Initialize the FastAPI application
app = FastAPI()


# route http posts to this method
@app.post('/')
async def test(request: Request):
    r = await request.body()
    # convert bytes of image data to uint8
    nparr = np.frombuffer(r, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # do some fancy processing here....
    prediction = XG_predict(img)
    # build a response dict to send back to client
    response = {'message': '{}'.format(prediction)}
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(content=response_pickled, status_code=200, media_type="application/json")


# route for the root URL
@app.get('/')
async def root():
    return 'Welcome to the FastAPI of the Animals Classification'


# start uvicorn server
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)