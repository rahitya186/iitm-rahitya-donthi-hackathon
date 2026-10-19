"""REST API: every endpoint, including input validation."""

import pandas as pd
from fastapi.testclient import TestClient

import api
import rebalancer
from conftest import T0


def test_api_endpoints(published):
    rebalancer.update(now=pd.Timestamp(T0) + pd.Timedelta(hours=6))

    with TestClient(api.app) as client:
        assert client.get("/health").json()["signals"] == len(published)

        rows = client.get("/signals", params={"ticker": "aapl"}).json()
        assert rows
        assert all(row["ticker"] == "AAPL" for row in rows)
        assert client.get("/signals", params={"since": "2026-10-10"}).status_code == 200  # date without timezone
        assert client.get("/signals", params={"since": "not-a-date"}).status_code == 400
        assert client.get("/signals", params={"min_impact": 11}).status_code == 422

        assert client.get("/companies").status_code == 200
        assert client.get("/companies/BA").json()["summary"]["ticker"] == "BA"
        assert client.get("/companies/XYZ").status_code == 404
        assert client.get("/events", params={"min_impact": 1}).status_code == 200

        result = client.post("/analyze", json={"text": "Moody's downgrades Boeing to junk"}).json()
        assert result["event_type"] == "Credit Event"
        assert result["companies"][0]["ticker"] == "BA"
        assert client.post("/analyze", json={"text": "hello", "source_type": "tv"}).status_code == 422
        assert client.post("/analyze", json={"text": "   "}).status_code == 400

        assert len(client.get("/index/weights").json()) == 15
        assert client.get("/index/history", params={"ticker": "ba"}).status_code == 200
