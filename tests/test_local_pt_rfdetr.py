"""
Integration tests for RF-DETR support in LocalPT module.
"""

import os
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from src.detection.local_pt_inference import LocalPT


class TestLocalPTRFDETR:
    """Test RF-DETR functionality in LocalPT."""
    
    def test_rfdetr_model_detection(self):
        """Test that RF-DETR models are correctly detected."""
        # Mock torch.load to return RF-DETR-like checkpoint
        mock_checkpoint = {
            'model': {
                'transformer.encoder.layers.0.weight': 'dummy',
                'decoder.layers.0.weight': 'dummy',
                'query_embed.weight': 'dummy'
            }
        }
        
        with patch('torch.load', return_value=mock_checkpoint), \
             patch('os.path.exists', return_value=True), \
             patch('src.detection.local_pt_inference.RFDETR_AVAILABLE', True), \
             patch('src.detection.local_pt_inference.RFDETRNano') as mock_rfdetr:
            
            mock_model = MagicMock()
            mock_rfdetr.return_value = mock_model
            
            local_pt = LocalPT('dummy_rfdetr.pt')
            
            assert local_pt.model_type == 'rf-detr'
            assert local_pt.model == mock_model
            mock_rfdetr.assert_called_once_with(pretrain_weights='dummy_rfdetr.pt')
    
    def test_yolo_model_fallback(self):
        """Test that YOLO models are still loaded correctly."""
        # Mock torch.load to return YOLO-like checkpoint
        mock_checkpoint = {
            'model': {
                'backbone.conv1.weight': 'dummy',
                'head.conv.weight': 'dummy'
            }
        }
        
        with patch('torch.load', return_value=mock_checkpoint), \
             patch('os.path.exists', return_value=True), \
             patch('src.detection.local_pt_inference.YOLO') as mock_yolo:
            
            mock_model = MagicMock()
            mock_yolo.return_value = mock_model
            
            local_pt = LocalPT('dummy_yolo.pt')
            
            assert local_pt.model_type == 'yolo'
            assert local_pt.model == mock_model
            mock_yolo.assert_called_once_with('dummy_yolo.pt')
    
    def test_rfdetr_inference_flow(self):
        """Test RF-DETR inference returns correct format."""
        # Mock RF-DETR model and results
        mock_results = MagicMock()
        mock_results.boxes = [[10, 20, 50, 60]]
        mock_results.labels = [0]
        mock_results.scores = [0.8]
        
        mock_model = MagicMock()
        mock_model.return_value = mock_results
        
        # Create test image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        with patch('torch.load', return_value={'model': {'transformer.weight': 'dummy'}}), \
             patch('os.path.exists', return_value=True), \
             patch('src.detection.local_pt_inference.RFDETR_AVAILABLE', True), \
             patch('src.detection.local_pt_inference.RFDETRNano', return_value=mock_model):
            
            local_pt = LocalPT('dummy_rfdetr.pt')
            result = local_pt.predict(test_image)
            
            # Verify standard format
            assert 'predictions' in result
            assert 'image' in result
            assert 'model' in result
            assert result['image']['width'] == 100
            assert result['image']['height'] == 100
    
    def test_model_info_rfdetr(self):
        """Test model info returns correct framework for RF-DETR."""
        with patch('torch.load', return_value={'model': {'transformer.weight': 'dummy'}}), \
             patch('os.path.exists', return_value=True), \
             patch('src.detection.local_pt_inference.RFDETR_AVAILABLE', True), \
             patch('src.detection.local_pt_inference.RFDETRNano'):
            
            local_pt = LocalPT('dummy_rfdetr.pt')
            info = local_pt.get_model_info()
            
            assert info['framework'] == 'rf-detr'
            assert info['type'] == 'pytorch'
    
    def test_rfdetr_unavailable_fallback(self):
        """Test graceful fallback when RF-DETR package is unavailable."""
        with patch('torch.load', return_value={'model': {'transformer.weight': 'dummy'}}), \
             patch('os.path.exists', return_value=True), \
             patch('src.detection.local_pt_inference.RFDETR_AVAILABLE', False):
            
            with pytest.raises(ImportError, match="RF-DETR package not available"):
                LocalPT('dummy_rfdetr.pt')