from locust import HttpUser, task, between
import json

SAMPLE_RFP_PAYLOAD = {
    "model": "rfp-extractor",
    "messages": [
        {"role": "system", "content": "You are an RFP information extraction engine. Return valid JSON."},
        {"role": "user", "content": "Extract buyer legal name and deadline from this sample tender text: 'The City of Medicine Hat issues this RFP 24-04 for ERP software. Proposals due Nov 15, 2026 at 2:00 PM MST.'"}
    ],
    "temperature": 0.0,
    "max_tokens": 512
}

class RFPModelUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def test_extraction_latency(self):
        headers = {"Content-Type": "application/json"}
        with self.client.post("/v1/chat/completions", data=json.dumps(SAMPLE_RFP_PAYLOAD), headers=headers, catch_response=True) as response:
            if response.status_code == 200 and "choices" in response.json():
                response.success()
            else:
                response.failure(f"Failed with status {response.status_code}: {response.text[:100]}")
