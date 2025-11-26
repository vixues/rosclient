"""Example usage of MockRosClient playback functionality."""
import time
from rosclient.clients import MockRosClient

def main():
    print("=" * 60)
    print("MockRosClient Playback Example")
    print("=" * 60)
    
    # Example 1: Create MockRosClient with playback file
    print("\n1. Creating MockRosClient with playback file...")
    client = MockRosClient(
        "mock://playback",
        config={
            "playback_file": "example_recording.rosrec",  # Path to recording file
            "playback_speed": 1.0,  # Real-time playback
            "playback_loop": False  # Don't loop
        }
    )
    
    # Connect (automatically starts playback if playback_file is set)
    client.connect_async()
    time.sleep(0.5)  # Wait for playback to start
    
    # Check if in playback mode
    if client.is_playback_mode():
        print("✓ Playback mode enabled")
        print(f"  Recording duration: {client._playback_recorder._metadata.total_duration:.2f}s")
        print(f"  Images: {client._playback_recorder._metadata.image_count}")
        print(f"  Point clouds: {client._playback_recorder._metadata.pointcloud_count}")
        print(f"  States: {client._playback_recorder._metadata.state_count}")
    else:
        print("✗ Playback mode not enabled (file not found or invalid)")
        print("  Falling back to mock data generation")
        return
    
    # Example 2: Access data as if it's a real ROS client
    print("\n2. Accessing data from playback...")
    for i in range(10):
        # Get latest image
        image_result = client.get_latest_image()
        if image_result:
            image, timestamp = image_result
            print(f"  Image {i+1}: shape={image.shape}, timestamp={timestamp:.3f}")
        
        # Get latest point cloud
        pc_result = client.get_latest_point_cloud()
        if pc_result:
            points, timestamp = pc_result
            print(f"  Point cloud {i+1}: {len(points)} points, timestamp={timestamp:.3f}")
        
        # Get current state
        state = client.get_status()
        print(f"  State {i+1}: mode={state.mode}, battery={state.battery:.1f}%, "
              f"pos=({state.latitude:.4f}, {state.longitude:.4f}, {state.altitude:.2f})")
        
        # Get playback progress
        progress = client.playback_get_progress()
        current_time = client.playback_get_current_time()
        print(f"  Playback: {progress*100:.1f}% complete, time={current_time:.2f}s")
        
        time.sleep(0.5)
    
    # Example 3: Playback control
    print("\n3. Testing playback controls...")
    
    # Pause playback
    print("  Pausing playback...")
    client.playback_pause()
    time.sleep(1)
    print(f"  Is paused: {client.playback_is_paused()}")
    
    # Resume playback
    print("  Resuming playback...")
    client.playback_resume()
    time.sleep(1)
    print(f"  Is playing: {client.playback_is_playing()}")
    
    # Change playback speed
    print("  Setting playback speed to 2x...")
    client.playback_set_speed(2.0)
    time.sleep(2)
    
    # Seek to specific time
    print("  Seeking to 5 seconds...")
    client.playback_seek(5.0)
    time.sleep(1)
    print(f"  Current time: {client.playback_get_current_time():.2f}s")
    
    # Get statistics
    stats = client.playback_get_statistics()
    if stats:
        print(f"  Statistics: {stats['images_played']} images, "
              f"{stats['pointclouds_played']} point clouds, "
              f"{stats['states_played']} states played")
    
    # Example 4: Continue accessing data
    print("\n4. Continuing to access data...")
    for i in range(5):
        image_result = client.get_latest_image()
        if image_result:
            image, timestamp = image_result
            print(f"  Image: shape={image.shape}")
        
        progress = client.playback_get_progress()
        print(f"  Progress: {progress*100:.1f}%")
        time.sleep(0.3)
    
    # Example 5: Stop playback
    print("\n5. Stopping playback...")
    client.playback_stop()
    print("  Playback stopped")
    
    # Cleanup
    client.terminate()
    print("\n✓ Example completed!")

if __name__ == "__main__":
    # Note: This example requires a recording file named "example_recording.rosrec"
    # You can create one using example_recording.py first
    print("Note: This example requires a recording file.")
    print("Run example_recording.py first to create 'example_recording.rosrec'")
    print()
    
    try:
        main()
    except FileNotFoundError:
        print("\n✗ Recording file not found!")
        print("  Please run example_recording.py first to create a recording file.")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

