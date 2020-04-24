### Micro Service
Here we create a micro service for our inference pipeline to use as endpoint. We create a flask API that takes in json input and generates a sentiment score for the input, we will use the model we trained using trainig_pipeline.py to get this job done.
We then dockerize the said API.

#### Run Instructions

1. `docker build -t MicroService:latest .` -- this references the `Dockerfile` at `.` (current directory) to build our Docker image & tags the docker image with `MicroService:latest`

2. Run `docker images` & find the newly built Docker image

3. `docker run -d -p 5000:5000 MicroService:latest` 

If everything worked properly, you should now have a container running, which:

1. Spins up a Flask server that accepts POST requests at http://0.0.0.0:5000/predict

2. Runs a sentiment classifier on the "data" field of the request 

3. Returns a response with the model's prediction 

To test this, you can either:

Write a POST request (e.g. using Postman or curl), here is an example response:

![alt text]()

This is how our model looks like:

![alt text]()


For further instructions refer the MakeFile in the MicroService folder.
