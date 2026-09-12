"""Test API endpoint for Convolutional Layers demonstration."""

import sys
import requests
import json


def test_convolutional_layers_endpoint():
    """Test the /convolutional-layers/demo endpoint."""
    print("Testing Convolutional Layers API endpoint...")
    print("=" * 60)

    # Base URL - adjust if your server runs on a different port
    base_url = "http://localhost:8000/api/deep-learning"

    # Test data
    test_request = {
        "num_filters": 32,
        "kernel_size": 3,
        "stride": 1,
        "padding": "same",
        "activation": "relu",
        "random_state": 42
    }

    try:
        # Test 1: POST request to /convolutional-layers/demo
        print("\n1. Testing POST /convolutional-layers/demo...")
        response = requests.post(
            f"{base_url}/convolutional-layers/demo",
            json=test_request,
            timeout=30
        )

        print(f"   Status code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"   Success: {data['success']}")
            print(f"   Execution time: {data['execution_time_ms']:.2f}ms")
            print(f"   Input image shape: {len(data['input_image'])}x{len(data['input_image'][0])}")
            print(f"   Number of filter kernels: {len(data['filter_kernels'])}")
            print(f"   Number of feature maps: {len(data['feature_maps'])}")
            print(f"   Output dimensions: {data['output_dimensions']}")
            print(f"   Common filters: {list(data['common_filters'].keys())}")
            print("   PASS")
        else:
            print(f"   FAIL: {response.text}")
            return False

        # Test 2: GET request to /convolutional-layers/info
        print("\n2. Testing GET /convolutional-layers/info...")
        response = requests.get(f"{base_url}/convolutional-layers/info", timeout=10)

        print(f"   Status code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            metadata = data['metadata']
            print(f"   Algorithm ID: {metadata['id']}")
            print(f"   Name: {metadata['name']}")
            print(f"   Category: {metadata['category']}")
            print(f"   Difficulty: {metadata['difficulty']}")
            print(f"   Parameters: {len(metadata['parameters'])}")
            print(f"   Use cases: {len(metadata['use_cases'])}")
            print("   PASS")
        else:
            print(f"   FAIL: {response.text}")
            return False

        # Test 3: Test with different parameters
        print("\n3. Testing with different parameters...")
        test_cases = [
            {"kernel_size": 5, "stride": 2, "padding": "valid"},
            {"num_filters": 16, "activation": "tanh"},
            {"kernel_size": 7, "padding": "same", "activation": "none"}
        ]

        for i, params in enumerate(test_cases, 1):
            request_data = {**test_request, **params}
            response = requests.post(
                f"{base_url}/convolutional-layers/demo",
                json=request_data,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                print(f"   Case {i} (params: {params}): "
                      f"output={data['output_dimensions']['height']}x{data['output_dimensions']['width']}, "
                      f"time={data['execution_time_ms']:.2f}ms - PASS")
            else:
                print(f"   Case {i}: FAIL - {response.status_code}")
                return False

        print("\n" + "=" * 60)
        print("ALL API TESTS PASSED!")
        print("=" * 60)
        return True

    except requests.exceptions.ConnectionError:
        print("\nERROR: Could not connect to the API server.")
        print("Make sure the server is running on http://localhost:8000")
        return False
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_algorithm_listing():
    """Test that the algorithm appears in the listing."""
    print("\n4. Testing algorithm listing...")
    base_url = "http://localhost:8000/api/deep-learning"

    try:
        response = requests.get(f"{base_url}/algorithms", timeout=10)

        if response.status_code == 200:
            algorithms = response.json()
            conv_layers = [a for a in algorithms if a['slug'] == 'convolutional-layers']

            if conv_layers:
                print(f"   Found 'convolutional-layers' in algorithm list")
                print(f"   Name: {conv_layers[0]['name']}")
                print("   PASS")
                return True
            else:
                print("   FAIL: 'convolutional-layers' not found in algorithm list")
                return False
        else:
            print(f"   FAIL: Status {response.status_code}")
            return False

    except Exception as e:
        print(f"   ERROR: {e}")
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CONVOLUTIONAL LAYERS API TEST")
    print("=" * 60)

    success = test_convolutional_layers_endpoint()

    if success:
        test_algorithm_listing()

    sys.exit(0 if success else 1)
