// k6 load test for the coalmine API (external-runner equivalent of
// coalmine/experiments/http_load.py):
//   python -m coalmine.api.server &
//   k6 run deploy/k6-load.js
import http from "k6/http";
import { check } from "k6";

export const options = {
  stages: [
    { duration: "15s", target: 32 },
    { duration: "45s", target: 32 },
    { duration: "10s", target: 0 },
  ],
  thresholds: {
    http_req_duration: ["p(99)<100"],
    http_req_failed: ["rate<0.001"],
  },
};

export default function () {
  const response = http.post(
    "http://127.0.0.1:8123/serve",
    JSON.stringify({}),
    { headers: { "Content-Type": "application/json" } }
  );
  check(response, {
    "status 200": (r) => r.status === 200,
    "has response": (r) => r.json("response") !== undefined,
  });
}
