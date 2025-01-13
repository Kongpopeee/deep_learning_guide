from locust import HttpUser, task, between
import json

class QuickstartUser(HttpUser):
    min_wait = 1000
    max_wait = 2000

    @task
    def test_api(self):

        data = {'x':-0.210738, 'y':-13.1719}
        self.client.post(
            url='/getclass',
            data=json.dumps(data),
            auth=('nuttachot', 'password')
            
        )
        