from inference_sdk import InferenceHTTPClient

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="VXketd8rJBvbWwf5E8EE"
)

# Using a known valid sample image URL
image_source = "https://media.roboflow.com/inference/cars.jpg"

result = client.run_workflow(
    workspace_name="aswin-gdjej",
    workflow_id="find-cars",
    images={
        "image": "https://c8.alamy.com/comp/2E4YCRF/aerial-view-of-a-parking-slot-with-parked-cars-2E4YCRF.jpg"
    },
    use_cache=True # cache workflow definition for 15 minutes
)

print(result) 