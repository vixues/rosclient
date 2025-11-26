import time
import pytest

from rosclient.clients.mock_client import MockRosClient


def test_mock_client_provides_images():
    client = MockRosClient("mock://test")

    # give background thread a moment to populate the image cache
    time.sleep(0.2)

    # fetch via fetch_camera_image (should return latest frame)
    frame_data = client.fetch_camera_image()
    assert frame_data is not None, "fetch_camera_image returned None"
    frame, ts = frame_data
    assert frame is not None
    assert hasattr(frame, "shape")

    # fetch via get_latest_image
    latest = client.get_latest_image()
    assert latest is not None, "get_latest_image returned None"
    f2, ts2 = latest
    assert f2 is not None

    # basic timestamp sanity
    assert ts2 >= 0
