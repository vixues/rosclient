"""Example usage of the recording module."""
import time
from rosclient.clients import MockRosClient
from rosclient.core import Recorder, RecordPlayer

def main():
    # Create a mock client
    client = MockRosClient("mock://test")
    client.connect_async()
    
    # Wait for connection
    time.sleep(1)
    
    print("Starting recording...")
    # Start recording
    client.start_recording(
        record_images=True,
        record_pointclouds=True,
        record_states=True,
        image_quality=85
    )
    
    # Simulate some data collection
    print("Collecting data for 5 seconds...")
    for i in range(50):
        image = client.get_latest_image()
        pointcloud = client.get_latest_point_cloud()
        state = client.get_status()
        
        if i % 10 == 0:
            stats = client.get_recording_statistics()
            print(f"  Recorded: {stats['images_recorded']} images, "
                  f"{stats['pointclouds_recorded']} point clouds, "
                  f"{stats['states_recorded']} states")
        
        time.sleep(0.1)
    
    # Stop recording
    print("Stopping recording...")
    client.stop_recording()
    
    # Save recording
    print("Saving recording...")
    client.save_recording("example_recording.rosrec", compress=True)
    
    # Load and play recording
    print("Loading recording...")
    recorder = Recorder.load("example_recording.rosrec")
    
    if recorder:
        print(f"Loaded recording: {recorder._metadata.total_duration:.2f}s, "
              f"{recorder._metadata.image_count} images, "
              f"{recorder._metadata.pointcloud_count} point clouds")
        
        # Create player
        player = RecordPlayer(recorder, playback_speed=1.0, loop=False)
        
        # Set callbacks
        image_count = [0]
        pointcloud_count = [0]
        state_count = [0]
        
        def on_image(image, timestamp):
            image_count[0] += 1
            if image_count[0] % 10 == 0:
                print(f"  Played image {image_count[0]}: shape={image.shape}")
        
        def on_pointcloud(points, timestamp):
            pointcloud_count[0] += 1
            if pointcloud_count[0] % 10 == 0:
                print(f"  Played point cloud {pointcloud_count[0]}: {len(points)} points")
        
        def on_state(state, timestamp):
            state_count[0] += 1
            if state_count[0] % 10 == 0:
                print(f"  Played state {state_count[0]}: mode={state.mode}, battery={state.battery:.1f}%")
        
        player.set_image_callback(on_image)
        player.set_pointcloud_callback(on_pointcloud)
        player.set_state_callback(on_state)
        
        # Play
        print("Playing recording...")
        player.play()
        
        # Wait for playback
        while player.is_playing():
            time.sleep(0.1)
            progress = player.get_progress()
            if progress > 0:
                print(f"  Progress: {progress*100:.1f}%")
        
        print("Playback completed!")
        print(f"Total played: {image_count[0]} images, "
              f"{pointcloud_count[0]} point clouds, "
              f"{state_count[0]} states")
    
    # Cleanup
    client.terminate()
    print("Done!")

if __name__ == "__main__":
    main()

