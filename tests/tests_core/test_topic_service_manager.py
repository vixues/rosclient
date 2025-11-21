"""Tests for TopicServiceManager."""
import pytest
from unittest.mock import Mock, MagicMock, patch
import logging

from rosclient.core.topic_service_manager import TopicServiceManager


class TestTopicServiceManager:
    """Tests for TopicServiceManager."""

    @pytest.fixture
    def mock_ros(self):
        """Create a mock roslibpy.Ros instance."""
        return Mock()

    @pytest.fixture
    def manager(self, mock_ros):
        """Create a TopicServiceManager instance."""
        return TopicServiceManager(mock_ros, "test_conn")

    def test_initialization(self, mock_ros):
        """Test TopicServiceManager initialization."""
        manager = TopicServiceManager(mock_ros, "test_conn")
        assert manager._ros is mock_ros
        assert len(manager._topics) == 0
        assert len(manager._services) == 0
        assert manager.log is not None

    def test_topic_creation(self, manager, mock_ros):
        """Test topic creation."""
        with patch('rosclient.core.topic_service_manager.roslibpy.Topic') as mock_topic_class:
            mock_topic = Mock()
            mock_topic_class.return_value = mock_topic
            
            topic = manager.topic("/test/topic", "std_msgs/String")
            
            assert topic is mock_topic
            mock_topic_class.assert_called_once_with(mock_ros, "/test/topic", "std_msgs/String")

    def test_topic_reuse(self, manager, mock_ros):
        """Test that same topic is reused."""
        with patch('rosclient.core.topic_service_manager.roslibpy.Topic') as mock_topic_class:
            mock_topic = Mock()
            mock_topic_class.return_value = mock_topic
            
            topic1 = manager.topic("/test/topic", "std_msgs/String")
            topic2 = manager.topic("/test/topic", "std_msgs/Int32")  # Different type, same name
            
            assert topic1 is topic2
            assert mock_topic_class.call_count == 1  # Only created once

    def test_service_creation(self, manager, mock_ros):
        """Test service creation."""
        with patch('rosclient.core.topic_service_manager.roslibpy.Service') as mock_service_class:
            mock_service = Mock()
            mock_service_class.return_value = mock_service
            
            service = manager.service("/test/service", "std_srvs/Empty")
            
            assert service is mock_service
            mock_service_class.assert_called_once_with(mock_ros, "/test/service", "std_srvs/Empty")

    def test_service_reuse(self, manager, mock_ros):
        """Test that same service is reused."""
        with patch('rosclient.core.topic_service_manager.roslibpy.Service') as mock_service_class:
            mock_service = Mock()
            mock_service_class.return_value = mock_service
            
            service1 = manager.service("/test/service", "std_srvs/Empty")
            service2 = manager.service("/test/service", "std_srvs/Trigger")
            
            assert service1 is service2
            assert mock_service_class.call_count == 1

    def test_close_all_topics(self, manager, mock_ros):
        """Test closing all topics."""
        with patch('rosclient.core.topic_service_manager.roslibpy.Topic') as mock_topic_class:
            mock_topic1 = Mock()
            mock_topic2 = Mock()
            mock_topic_class.side_effect = [mock_topic1, mock_topic2]
            
            manager.topic("/topic1", "")
            manager.topic("/topic2", "")
            
            assert len(manager._topics) == 2
            
            manager.close_all()
            
            assert mock_topic1.unsubscribe.called
            assert mock_topic2.unsubscribe.called
            assert len(manager._topics) == 0

    def test_close_all_services(self, manager, mock_ros):
        """Test closing all services."""
        with patch('rosclient.core.topic_service_manager.roslibpy.Service') as mock_service_class:
            mock_service1 = Mock()
            mock_service1.unadvertise = Mock()
            mock_service2 = Mock()
            mock_service2.unadvertise = Mock()
            mock_service_class.side_effect = [mock_service1, mock_service2]
            
            manager.service("/service1", "")
            manager.service("/service2", "")
            
            assert len(manager._services) == 2
            
            manager.close_all()
            
            assert mock_service1.unadvertise.called
            assert mock_service2.unadvertise.called
            assert len(manager._services) == 0

    def test_close_all_handles_exceptions(self, manager, mock_ros):
        """Test that close_all handles exceptions gracefully."""
        with patch('rosclient.core.topic_service_manager.roslibpy.Topic') as mock_topic_class:
            mock_topic = Mock()
            mock_topic.unsubscribe.side_effect = Exception("Test error")
            mock_topic_class.return_value = mock_topic
            
            manager.topic("/test/topic", "")
            manager.close_all()  # Should not raise
            
            assert len(manager._topics) == 0  # Still cleared

    def test_logger_level(self, mock_ros):
        """Test custom logger level."""
        manager = TopicServiceManager(mock_ros, "test_conn", logger_level=logging.WARNING)
        assert manager.log.level == logging.WARNING

